use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use fusio::{
    disk::LocalFs,
    executor::{NoopExecutor, tokio::TokioExecutor},
    mem::fs::InMemoryFs,
    path::Path,
};

use super::common::workspace_temp_dir;
use crate::{
    compaction::{
        executor::LocalCompactionExecutor,
        planner::{LeveledCompactionPlanner, LeveledPlannerConfig},
    },
    db::{DB, Expr, ScalarValue},
    inmem::policy::BatchesThreshold,
    mode::TONBO_FIELD_ID_META_KEY,
    ondisk::sstable::{SsTableConfig, SsTableDescriptor, SsTableId},
    schema::SchemaBuilder,
};

fn field_with_id(name: &str, data_type: DataType, nullable: bool, field_id: u32) -> Field {
    let mut meta = HashMap::new();
    meta.insert(TONBO_FIELD_ID_META_KEY.to_string(), field_id.to_string());
    Field::new(name, data_type, nullable).with_metadata(meta)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn online_schema_evolution_across_sst_and_memtable_with_null_backfill() {
    // v1 schema: [id (#1, pk), speed (#2, f64)]
    let schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Float64, true, 2),
    ]));
    let config_v1 = SchemaBuilder::from_schema(Arc::clone(&schema_v1))
        .primary_key("id")
        .build()
        .expect("v1 config")
        .with_schema_version(1);

    let executor = Arc::new(NoopExecutor);
    let db = DB::<InMemoryFs, NoopExecutor>::new(config_v1, Arc::clone(&executor))
        .await
        .expect("new db");
    assert_eq!(db.schema_version(), 1);

    // 1. Ingest v1 rows and flush to an older SSTable (SST #1 only physically has columns #1, #2).
    let batch_v1 = RecordBatch::try_new(
        Arc::clone(&schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["train-1", "train-2"])),
            Arc::new(Float64Array::from(vec![Some(120.0), Some(150.0)])),
        ],
    )
    .expect("batch_v1");
    db.ingest(batch_v1).await.expect("ingest v1");
    if let Some(seg) = db.inner().seal_mutable() {
        db.inner().add_immutable(seg, None);
    }
    let sst_root = Path::from("schema_evolution_sst");
    let sst_cfg = Arc::new(
        SsTableConfig::new(
            Arc::clone(&schema_v1),
            Arc::clone(&db.inner().fs),
            sst_root.clone(),
        )
        .with_key_extractor(Arc::clone(db.inner().extractor())),
    );
    db.inner()
        .flush_immutables_with_descriptor(
            Arc::clone(&sst_cfg),
            SsTableDescriptor::new(SsTableId::new(1), 0),
        )
        .await
        .expect("flush v1 sst");

    // 2. Evolve schema online to v2: add `resistance` (#3, f64, nullable) and `limit_speed` (#4, i32, nullable).
    let schema_v2: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Float64, true, 2),
        field_with_id("resistance", DataType::Float64, true, 3),
        field_with_id("limit_speed", DataType::Int32, true, 4),
    ]));
    let updated_meta = db
        .evolve_schema(Arc::clone(&schema_v2), 2)
        .await
        .expect("evolve to v2");
    assert_eq!(updated_meta.schema_version, 2);
    assert_eq!(db.schema_version(), 2);
    assert_eq!(db.schema().fields().len(), 4);

    // 3. Ingest both a v1-shaped subset batch and a full v2 batch into the SAME mutable memtable!
    let batch_v1_in_v2 = RecordBatch::try_new(
        Arc::clone(&schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["train-3"])),
            Arc::new(Float64Array::from(vec![Some(180.0)])),
        ],
    )
    .expect("batch_v1_in_v2");
    db.ingest(batch_v1_in_v2)
        .await
        .expect("ingest v1 subset batch after v2 evolution");

    let batch_v2 = RecordBatch::try_new(
        Arc::clone(&schema_v2),
        vec![
            Arc::new(StringArray::from(vec!["train-2", "train-4"])),
            Arc::new(Float64Array::from(vec![Some(160.0), Some(200.0)])),
            Arc::new(Float64Array::from(vec![Some(12.5), None])),
            Arc::new(Int32Array::from(vec![Some(250), Some(300)])),
        ],
    )
    .expect("batch_v2");
    db.ingest(batch_v2).await.expect("ingest v2 batch");

    // 4. Scan full union schema across old SSTable (v1) + Mutable MemTable (mixed v1 & v2).
    let batches = db.scan().collect().await.expect("full scan");
    let mut rows = Vec::new();
    for batch in &batches {
        assert_eq!(batch.schema().fields().len(), 4);
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id");
        let speeds = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("speed");
        let resistances = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("resistance");
        let limits = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("limit_speed");
        for i in 0..batch.num_rows() {
            rows.push((
                ids.value(i).to_string(),
                (!speeds.is_null(i)).then(|| speeds.value(i)),
                (!resistances.is_null(i)).then(|| resistances.value(i)),
                (!limits.is_null(i)).then(|| limits.value(i)),
            ));
        }
    }
    assert_eq!(
        rows,
        vec![
            // train-1 comes from the old v1 SSTable -> resistance & limit_speed are backfilled as NULL!
            ("train-1".to_string(), Some(120.0), None, None),
            // train-2 was updated in v2 MemTable -> shadows v1 SSTable entry
            ("train-2".to_string(), Some(160.0), Some(12.5), Some(250)),
            // train-3 was ingested with v1 subset schema into v2 MemTable -> NULL for v2 columns
            ("train-3".to_string(), Some(180.0), None, None),
            // train-4 was ingested with v2 schema
            ("train-4".to_string(), Some(200.0), None, Some(300)),
        ]
    );

    // 5. Predicate filtering on a v2-only column (`limit_speed > 260`) across old v1 SSTable and v2 MemTable.
    let filtered = db
        .scan()
        .filter(Expr::gt("limit_speed", ScalarValue::from(260i32)))
        .collect()
        .await
        .expect("filtered scan on v2 column");
    let filtered_ids: Vec<String> = filtered
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("id");
            (0..b.num_rows()).map(move |i| col.value(i).to_string())
        })
        .collect();
    assert_eq!(filtered_ids, vec!["train-4".to_string()]);

    // 6. Predicate `IS NULL` on a v2-only column should return train-1 (from old v1 SSTable) and train-3!
    let null_filtered = db
        .scan()
        .filter(Expr::is_null("limit_speed"))
        .collect()
        .await
        .expect("is_null scan on v2 column");
    let null_ids: Vec<String> = null_filtered
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("id");
            (0..b.num_rows()).map(move |i| col.value(i).to_string())
        })
        .collect();
    assert_eq!(null_ids, vec!["train-1".to_string(), "train-3".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn field_id_prevents_dropped_and_readded_same_name_column_collision() {
    // v1: [id (#1), metric (#2, i32)]
    let schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("metric", DataType::Int32, true, 2),
    ]));
    let config_v1 = SchemaBuilder::from_schema(Arc::clone(&schema_v1))
        .primary_key("id")
        .build()
        .expect("v1 config")
        .with_schema_version(1);

    let executor = Arc::new(NoopExecutor);
    let db = DB::<InMemoryFs, NoopExecutor>::new(config_v1, Arc::clone(&executor))
        .await
        .expect("new db");

    // Ingest row into v1 and flush to SST #1.
    let batch_v1 = RecordBatch::try_new(
        Arc::clone(&schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["k1"])),
            Arc::new(Int32Array::from(vec![Some(999)])),
        ],
    )
    .expect("batch_v1");
    db.ingest(batch_v1).await.expect("ingest v1");
    if let Some(seg) = db.inner().seal_mutable() {
        db.inner().add_immutable(seg, None);
    }
    let sst_root = Path::from("field_id_collision_sst");
    let sst_cfg = Arc::new(
        SsTableConfig::new(Arc::clone(&schema_v1), Arc::clone(&db.inner().fs), sst_root)
            .with_key_extractor(Arc::clone(db.inner().extractor())),
    );
    db.inner()
        .flush_immutables_with_descriptor(sst_cfg, SsTableDescriptor::new(SsTableId::new(1), 0))
        .await
        .expect("flush v1 sst");

    // Suppose in v2 `metric` (#2) was logically dropped, and in v3 a brand-new `metric` (#3, i32)
    // was added! When we evolve the table to v3 with `metric` carrying field_id = 3, reading `k1`
    // from the v1 SSTable MUST return NULL for `metric` (#3), NOT 999 from `metric` (#2)!
    let schema_v3: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("metric", DataType::Int32, true, 3),
    ]));
    db.evolve_schema(Arc::clone(&schema_v3), 3)
        .await
        .expect("evolve to v3");

    let batches = db
        .scan()
        .projection(Arc::clone(&schema_v3))
        .collect()
        .await
        .expect("scan v3 projection");
    assert_eq!(batches.len(), 1);
    let metric_col = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("metric col");
    assert!(
        metric_col.is_null(0),
        "old SSTable column with field_id=2 must not be conflated with new column of same name with field_id=3"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn major_compaction_merges_old_and_new_schema_sstables() {
    let root_dir = workspace_temp_dir("tonbo-schema-evolve-compaction");
    let schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
    ]));
    let config_v1 = SchemaBuilder::from_schema(Arc::clone(&schema_v1))
        .primary_key("id")
        .build()
        .expect("v1 config")
        .with_schema_version(1);

    let executor = Arc::new(TokioExecutor::default());
    let db: DB<LocalFs, TokioExecutor> = DB::<LocalFs, TokioExecutor>::builder(config_v1)
        .on_disk(&root_dir)
        .expect("on_disk")
        .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
        .with_minor_compaction(1, 0)
        .open_with_executor(Arc::clone(&executor))
        .await
        .expect("open db");

    // Flush SST #1 at schema v1
    let batch_v1 = RecordBatch::try_new(
        Arc::clone(&schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Int32Array::from(vec![Some(10), Some(20)])),
        ],
    )
    .expect("batch_v1");
    db.ingest(batch_v1).await.expect("ingest v1");

    // Evolve to schema v2 (adds `altitude` #3) and flush SST #2 at schema v2
    let schema_v2: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
        field_with_id("altitude", DataType::Int32, true, 3),
    ]));
    db.evolve_schema(Arc::clone(&schema_v2), 2)
        .await
        .expect("evolve to v2");

    let batch_v2 = RecordBatch::try_new(
        Arc::clone(&schema_v2),
        vec![
            Arc::new(StringArray::from(vec!["b", "c"])),
            Arc::new(Int32Array::from(vec![Some(25), Some(30)])),
            Arc::new(Int32Array::from(vec![Some(1000), Some(2000)])),
        ],
    )
    .expect("batch_v2");
    db.ingest(batch_v2).await.expect("ingest v2");

    // Run major compaction merging the L0 SSTs (one written with v1 schema, one with v2 schema)
    let planner_cfg = LeveledPlannerConfig {
        l0_trigger: 2,
        ..Default::default()
    };
    let planner = LeveledCompactionPlanner::new(planner_cfg);
    let sst_cfg = Arc::new(
        SsTableConfig::new(
            db.schema(),
            Arc::clone(&db.inner().fs),
            db.inner().sst_root.clone(),
        )
        .with_key_extractor(Arc::clone(db.inner().extractor())),
    );
    let compactor = LocalCompactionExecutor::new(sst_cfg, 10);
    let outcome = db
        .inner()
        .run_compaction_task(&planner, &compactor)
        .await
        .expect("major compaction should succeed across v1 and v2 SSTables");
    assert!(outcome.is_some(), "expected major compaction to run");

    let batches = db.scan().collect().await.expect("scan after compaction");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id");
        let speeds = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("speed");
        let altitudes = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("altitude");
        for i in 0..batch.num_rows() {
            rows.push((
                ids.value(i).to_string(),
                (!speeds.is_null(i)).then(|| speeds.value(i)),
                (!altitudes.is_null(i)).then(|| altitudes.value(i)),
            ));
        }
    }
    assert_eq!(
        rows,
        vec![
            ("a".to_string(), Some(10), None),
            ("b".to_string(), Some(25), Some(1000)),
            ("c".to_string(), Some(30), Some(2000)),
        ]
    );
    let _ = std::fs::remove_dir_all(root_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_table_catalog_independent_schemas_and_non_colliding_sst_ids() {
    let root_dir = workspace_temp_dir("tonbo-multi-table-catalog");
    let executor = Arc::new(TokioExecutor::default());

    let train_schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
    ]));
    let train_cfg = SchemaBuilder::from_schema(Arc::clone(&train_schema_v1))
        .primary_key("id")
        .build()
        .expect("train cfg")
        .with_schema_version(1);

    let train_db: DB<LocalFs, TokioExecutor> = DB::<LocalFs, TokioExecutor>::builder(train_cfg)
        .on_disk(&root_dir)
        .expect("on_disk")
        .table_name("train")
        .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
        .with_minor_compaction(1, 0)
        .open_with_executor(Arc::clone(&executor))
        .await
        .expect("open train db");

    let train_batch = RecordBatch::try_new(
        Arc::clone(&train_schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["CR400AF"])),
            Arc::new(Int32Array::from(vec![Some(350)])),
        ],
    )
    .expect("train batch");
    train_db.ingest(train_batch).await.expect("ingest train");

    // Open second table "airplane" on the same root directory
    let airplane_schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("wingspan_m", DataType::Int32, true, 2),
    ]));
    let airplane_cfg = SchemaBuilder::from_schema(Arc::clone(&airplane_schema_v1))
        .primary_key("id")
        .build()
        .expect("airplane cfg")
        .with_schema_version(1);

    let airplane_db: DB<LocalFs, TokioExecutor> =
        DB::<LocalFs, TokioExecutor>::builder(airplane_cfg)
            .on_disk(&root_dir)
            .expect("on_disk")
            .table_name("airplane")
            .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
            .with_minor_compaction(1, 0)
            .open_with_executor(Arc::clone(&executor))
            .await
            .expect("open airplane db");

    assert_ne!(
        train_db.table_id(),
        airplane_db.table_id(),
        "train and airplane must receive distinct TableIds in the catalog"
    );

    let airplane_batch = RecordBatch::try_new(
        Arc::clone(&airplane_schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["C919"])),
            Arc::new(Int32Array::from(vec![Some(36)])),
        ],
    )
    .expect("airplane batch");
    airplane_db
        .ingest(airplane_batch)
        .await
        .expect("ingest airplane");

    let tables = airplane_db
        .manifest()
        .list_tables()
        .await
        .expect("list tables");
    assert!(tables.contains(&train_db.table_id()));
    assert!(tables.contains(&airplane_db.table_id()));

    // Verify train's SSTable was not overwritten by airplane's SSTable (non-colliding SsTableIds)
    let train_batches = train_db.scan().collect().await.expect("scan train");
    assert_eq!(train_batches.len(), 1);
    let train_ids = train_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("train id");
    assert_eq!(train_ids.value(0), "CR400AF");

    let airplane_batches = airplane_db.scan().collect().await.expect("scan airplane");
    assert_eq!(airplane_batches.len(), 1);
    let airplane_ids = airplane_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("airplane id");
    assert_eq!(airplane_ids.value(0), "C919");

    let _ = std::fs::remove_dir_all(root_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wal_recovery_across_schema_evolution_and_guardrail_checks() {
    let root_dir = workspace_temp_dir("tonbo-wal-schema-evolve");
    let executor = Arc::new(TokioExecutor::default());

    let schema_v1: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
    ]));
    let cfg_v1 = SchemaBuilder::from_schema(Arc::clone(&schema_v1))
        .primary_key("id")
        .build()
        .expect("v1 cfg")
        .with_schema_version(1);

    let db: DB<LocalFs, TokioExecutor> = DB::<LocalFs, TokioExecutor>::builder(cfg_v1)
        .on_disk(&root_dir)
        .expect("on_disk")
        .open_with_executor(Arc::clone(&executor))
        .await
        .expect("open v1 db");

    // Ingest v1 row into WAL (without flushing to SST)
    let batch_v1 = RecordBatch::try_new(
        Arc::clone(&schema_v1),
        vec![
            Arc::new(StringArray::from(vec!["row-v1"])),
            Arc::new(Int32Array::from(vec![Some(100)])),
        ],
    )
    .expect("batch_v1");
    db.ingest(batch_v1).await.expect("ingest v1");

    // Guardrail 1: non-increasing schema_version must be rejected
    let schema_v2: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
        field_with_id("mass", DataType::Int32, true, 3),
    ]));
    assert!(
        db.evolve_schema(Arc::clone(&schema_v2), 1).await.is_err(),
        "same schema_version must be rejected"
    );

    // Guardrail 2: non-nullable new column must be rejected
    let bad_schema_non_nullable: SchemaRef = Arc::new(Schema::new(vec![
        field_with_id("id", DataType::Utf8, false, 1),
        field_with_id("speed", DataType::Int32, true, 2),
        field_with_id("mass", DataType::Int32, false, 3),
    ]));
    assert!(
        db.evolve_schema(bad_schema_non_nullable, 2).await.is_err(),
        "non-nullable new column must be rejected"
    );

    // Evolve online to v2 and ingest a v2 row into the same WAL
    db.evolve_schema(Arc::clone(&schema_v2), 2)
        .await
        .expect("evolve to v2");
    let batch_v2 = RecordBatch::try_new(
        Arc::clone(&schema_v2),
        vec![
            Arc::new(StringArray::from(vec!["row-v2"])),
            Arc::new(Int32Array::from(vec![Some(200)])),
            Arc::new(Int32Array::from(vec![Some(5000)])),
        ],
    )
    .expect("batch_v2");
    db.ingest(batch_v2).await.expect("ingest v2");
    db.into_inner()
        .disable_wal_async()
        .await
        .expect("disable wal");

    // Reopen with v2 schema and verify WAL recovery replays both v1 and v2 frames seamlessly!
    let cfg_v2_reopen = SchemaBuilder::from_schema(Arc::clone(&schema_v2))
        .primary_key("id")
        .build()
        .expect("v2 cfg")
        .with_schema_version(2);
    let recovered: DB<LocalFs, TokioExecutor> =
        DB::<LocalFs, TokioExecutor>::builder(cfg_v2_reopen)
            .on_disk(&root_dir)
            .expect("on_disk")
            .open_with_executor(Arc::clone(&executor))
            .await
            .expect("reopen v2 db");

    assert_eq!(recovered.schema_version(), 2);
    let batches = recovered.scan().collect().await.expect("scan recovered");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id");
        let speeds = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("speed");
        let masses = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("mass");
        for i in 0..batch.num_rows() {
            rows.push((
                ids.value(i).to_string(),
                (!speeds.is_null(i)).then(|| speeds.value(i)),
                (!masses.is_null(i)).then(|| masses.value(i)),
            ));
        }
    }
    assert_eq!(
        rows,
        vec![
            ("row-v1".to_string(), Some(100), None),
            ("row-v2".to_string(), Some(200), Some(5000)),
        ]
    );

    let _ = std::fs::remove_dir_all(root_dir);
}
