"""Database operation utilities for the egocentric_dataset_clips Supabase table.

This module mirrors the structure of motion_generation/code/rewriter/
database_operation_rewriter.py but targets the egocentric clips table
hosted on the new Supabase instance. It is intended to support the
conversion of Ego4D / Egocentric-100K data into LeRobot v3.0 format.

The egocentric data is *not* stored in two separate tables but rather in
one table `egocentric_dataset_clips` partitioned by the `partition`
column. The two partitions of interest are:

    - "Ego4D"
    - "Egocentric-100K"

Running this script directly performs a field-level statistical analysis
of both partitions (non-null ratios, status value distributions, caption
JSON sub-keys), which is useful for designing the LeRobot v3.0 schema
mapping.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any, Callable

from postgrest.exceptions import APIError
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Supabase configuration
# -----------------------------------------------------------------------------
SUPABASE_URL = "http://spb-uf6v6z0632cr24if.supabase-vpc.opentrust.net"
SUPABASE_KEY = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
    "eyJyb2xlIjoiYW5vbiIsInJlZiI6InNwYi11ZjZ2NnowNjMyY3IyNGlmIiwiaXNzIjoic3VwYWJhc2UiLCJpYXQiOjE3NzYwNzk2MzQsImV4cCI6MjA5MTY1NTYzNH0."
    "FZfGAyqsJYmLjJSWhnRc-U_T0oq_RMQqqsUrFVdZX-o"
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

EGOCENTRIC_DATASET_CLIPS_TABLE = "egocentric_dataset_clips"

# Logical partitions exposed by the table.
PARTITION_EGO4D = "Ego4D"
PARTITION_EGO100K = "Egocentric-100K"
EGOCENTRIC_PARTITIONS = (PARTITION_EGO4D, PARTITION_EGO100K)


def get_egocentric_columns() -> set[str]:
    """Return the known column names of egocentric_dataset_clips."""
    return {
        "id", "partition", "path", "dataset_name", "sub", "part",
        "source_id", "source_path", "create_time", "random_number",
        "duration", "fps", "height_resolution", "width_resolution",
        "start_frame", "end_frame",
        "video_tag", "hand_tag",
        "pose3d_hand_path", "pose3d_hand_status", "pose3d_hand_process_time",
        "agilex_ik_result_meta_data_path", "agilex_ik_result_data_path",
        "agilex_ik_status", "agilex_ik_process_time",
        "agilex_render_video_path", "agilex_render_status", "agilex_render_process_time",
        "caption", "caption_status", "caption_process_time",
        "motion_score", "motion_score_status",
        "track_info_path",
    }


# -----------------------------------------------------------------------------
# Generic helpers (kept compatible with database_operation_rewriter.py)
# -----------------------------------------------------------------------------

def _postgrest_error_summary(exc: BaseException) -> str:
    msg = getattr(exc, "message", None)
    if isinstance(msg, dict):
        parts = []
        for k in ("message", "details", "hint", "code"):
            v = msg.get(k)
            if v not in (None, ""):
                parts.append(f"{k}={v}")
        return "; ".join(parts) if parts else str(msg)
    if msg not in (None, ""):
        return str(msg)
    det = getattr(exc, "details", None)
    if det not in (None, ""):
        return str(det)
    hint = getattr(exc, "hint", None)
    if hint not in (None, ""):
        return str(hint)
    return f"{type(exc).__name__}"


def _is_statement_timeout(err: APIError) -> bool:
    code = str(getattr(err, "code", "") or "")
    if code == "57014":
        return True
    msg = (getattr(err, "message", None) or str(err)).lower()
    return "statement timeout" in msg


def _execute_order_path_limit_with_retry(
    build_filtered_query: Callable[[], Any],
    limit: int,
    *,
    max_retries: int = 8,
    min_limit: int = 20,
):
    """Run `build_filtered_query().order('path').limit(N).execute()` with
    exponential back-off on PostgreSQL statement_timeout (57014).
    """
    limit_cur = max(min_limit, int(limit))
    last_err: APIError | None = None
    for attempt in range(max_retries):
        try:
            return build_filtered_query().order("path").limit(limit_cur).execute()
        except APIError as e:
            last_err = e
            if not _is_statement_timeout(e) or attempt == max_retries - 1:
                raise
            limit_cur = max(min_limit, limit_cur // 2)
            delay = min(60.0, 2.0 ** (attempt + 1))
            logger.warning(
                "egocentric query statement timeout, retry in %.0fs (attempt %s/%s, limit=%s)",
                delay, attempt + 1, max_retries, limit_cur,
            )
            time.sleep(delay)
    assert last_err is not None
    raise last_err


# =============================================================================
# Query helpers for egocentric_dataset_clips
# =============================================================================

def get_egocentric_paths_from_database(
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    partition: str | None = None,
    dataset_name: str | None = None,
    total_num: int | None = None,
    path_gte: str | None = None,
    path_lt: str | None = None,
    select_fields: list[str] | None = None,
    filter_empty_fields: list[str] | None = None,
    require_empty_fields: list[str] | None = None,
    page_size: int = 80,
):
    """Cursor-style pagination over egocentric_dataset_clips ordered by `path`.

    Args:
        partition: filter by `partition` column (preferred for Ego4D / Egocentric-100K).
        dataset_name: filter by `dataset_name` column.
        total_num: cap on returned rows (None for all).
        path_gte / path_lt: half-open path range for sharding across machines.
        select_fields: explicit projection. Defaults to a compact set.
        filter_empty_fields: rows where listed fields are NOT NULL.
        require_empty_fields: rows where listed fields ARE NULL (eg. unprocessed).
        page_size: per-page limit, kept small to avoid statement_timeout.
    """
    if total_num is not None and total_num <= 0:
        raise ValueError("total_num must be > 0")
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    if select_fields is None:
        select_cols = "path,id,partition,dataset_name,sub,part"
    else:
        select_cols = ",".join(select_fields)

    default_page_size = page_size
    all_data: list[dict] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    last_path: str | None = None

    while total_num is None or len(all_data) < total_num:
        remaining = None if total_num is None else total_num - len(all_data)
        page_size = default_page_size if remaining is None else min(default_page_size, remaining)

        def build_filtered_query():
            q = supabase.table(table_name).select(select_cols)
            if partition:
                q = q.eq("partition", partition)
            if dataset_name:
                q = q.eq("dataset_name", dataset_name)
            if path_gte is not None:
                q = q.gte("path", path_gte)
            if path_lt is not None:
                q = q.lt("path", path_lt)
            if last_path is not None:
                q = q.gt("path", last_path)
            if filter_empty_fields:
                for field in filter_empty_fields:
                    q = q.not_.is_(field, "null")
            if require_empty_fields:
                for field in require_empty_fields:
                    q = q.is_(field, "null")
            return q

        response = _execute_order_path_limit_with_retry(build_filtered_query, page_size)
        page_data = response.data or []
        if not page_data:
            break

        for item in page_data:
            path = item.get("path")
            raw_id = item.get("id")
            if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                clip_id = os.path.splitext(os.path.basename(path or ""))[0]
            else:
                clip_id = str(raw_id).strip()
            if clip_id in seen_ids or path in seen_paths:
                continue
            seen_ids.add(clip_id)
            seen_paths.add(path)
            all_data.append(item)
            if total_num is not None and len(all_data) >= total_num:
                break

        if len(page_data) < page_size:
            break
        last_path = page_data[-1]["path"]

    class _Result:
        pass
    result = _Result()
    result.data = all_data
    return result


def get_egocentric_boundaries(
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    partition: str | None = None,
    total_machines: int = 10,
    sample_interval: int = 50000,
    page_size: int = 1000,
    verbose: bool = True,
) -> list[str]:
    """Sample boundary paths to shard the table across `total_machines` workers.

    Returns total_machines+1 boundary paths; worker i processes
    [boundaries[i], boundaries[i+1]).
    """
    sampled: list[str] = []
    last_path: str | None = None
    count_since_last = 0
    total_scanned = 0
    page_count = 0

    while True:
        def build_filtered_query():
            q = supabase.table(table_name).select("path")
            if partition:
                q = q.eq("partition", partition)
            if last_path is not None:
                q = q.gt("path", last_path)
            return q

        response = _execute_order_path_limit_with_retry(build_filtered_query, page_size)
        page_data = response.data or []
        if not page_data:
            break

        for item in page_data:
            p = item["path"]
            if count_since_last >= sample_interval or not sampled:
                sampled.append(p)
                count_since_last = 0
            count_since_last += 1
            last_path = p

        total_scanned += len(page_data)
        page_count += 1
        if verbose and page_count % 10 == 0:
            print(f"[egocentric_boundaries] scanned {total_scanned}, sampled {len(sampled)} boundaries")

        if len(page_data) < page_size:
            break

    if not sampled:
        return []

    n = len(sampled)
    if n <= total_machines + 1:
        boundaries = list(sampled)
    else:
        indices = (
            [0]
            + [int((i + 1) * (n - 1) / total_machines) for i in range(total_machines - 1)]
            + [n - 1]
        )
        boundaries = [sampled[i] for i in indices]

    boundaries[-1] = boundaries[-1] + "\uffff"
    return boundaries


# =============================================================================
# Single-row CRUD helpers (path is the logical primary key)
# =============================================================================

def supabase_select(
    unique_value: str,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    unique_key: str = "path",
):
    return (
        supabase.table(table_name)
        .select("*")
        .eq(unique_key, unique_value)
        .execute()
    )


def supabase_update(
    record: dict,
    unique_value: str,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    unique_key: str = "path",
):
    return (
        supabase.table(table_name)
        .update(record)
        .eq(unique_key, unique_value)
        .execute()
    )


def supabase_upsert(
    record: dict,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    on_conflict: str = "path",
):
    return (
        supabase.table(table_name)
        .upsert(record, on_conflict=on_conflict)
        .execute()
    )


def supabase_insert(
    record: dict,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
):
    return supabase.table(table_name).insert(record).execute()


def check_field_empty(
    field_name: str,
    path_value: str,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
) -> bool:
    response = (
        supabase.table(table_name)
        .select(field_name)
        .eq("path", path_value)
        .execute()
    )
    if not response.data:
        return True
    record = response.data[0]
    if field_name not in record:
        return True
    v = record[field_name]
    return v is None or v == ""


def batch_update(
    updates: list[dict],
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
    batch_size: int = 50,
):
    results = []
    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        for record in batch:
            path = record.get("path")
            if not path:
                logger.warning("skip record without path: %s", record)
                continue
            try:
                update_data = {k: v for k, v in record.items() if k != "path"}
                resp = supabase_update(update_data, path, table_name)
                results.append(resp)
            except Exception as e:
                logger.error("update failed path=%s: %s", path, _postgrest_error_summary(e))
    return results


# =============================================================================
# Field analysis utilities (for LeRobot v3.0 schema design)
# =============================================================================

def _try_parse_json(v: Any) -> Any:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def analyze_partition(
    partition: str,
    sample_size: int = 500,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
) -> dict:
    """Sample `sample_size` rows from a partition and report:

    - per-field non-null ratio + an example value
    - inferred Python type per field
    - distribution of *_status columns
    - JSON sub-keys for structured columns (caption, hand_tag)

    The returned dict is JSON-serialisable.
    """
    res = get_egocentric_paths_from_database(
        table_name=table_name,
        partition=partition,
        total_num=sample_size,
        select_fields=["*"],
        page_size=min(200, sample_size),
    )
    rows = res.data
    n = len(rows)
    if n == 0:
        return {"partition": partition, "sample_size": 0, "fields": {}}

    field_keys: set[str] = set()
    for r in rows:
        field_keys.update(r.keys())

    field_stats: dict[str, dict] = {}
    status_dists: dict[str, Counter] = {}
    json_subkeys: dict[str, Counter] = {}

    for k in sorted(field_keys):
        non_null = 0
        example: Any = None
        type_name = "NoneType"
        for r in rows:
            v = r.get(k)
            if v is None or v == "":
                continue
            non_null += 1
            if example is None:
                example = v
                type_name = type(v).__name__

        field_stats[k] = {
            "non_null_ratio": round(non_null / n, 4),
            "non_null_count": non_null,
            "type": type_name,
            "example": (str(example)[:200] if example is not None else None),
        }

        if k.endswith("_status"):
            status_dists[k] = Counter(r.get(k) for r in rows)

        for r in rows:
            parsed = _try_parse_json(r.get(k))
            if isinstance(parsed, dict):
                json_subkeys.setdefault(k, Counter()).update(parsed.keys())

    return {
        "partition": partition,
        "sample_size": n,
        "fields": field_stats,
        "status_distribution": {k: dict(c) for k, c in status_dists.items()},
        "json_subkeys": {k: dict(c) for k, c in json_subkeys.items()},
    }


def print_partition_report(report: dict) -> None:
    print(f"\n========== partition = {report['partition']!r}  (sample={report['sample_size']}) ==========")
    if report["sample_size"] == 0:
        print("  <empty>")
        return

    print("\n-- field non-null ratio / type / example --")
    width = max(len(k) for k in report["fields"])
    for k, s in report["fields"].items():
        ratio = f"{s['non_null_ratio']*100:6.2f}%"
        print(f"  {k.ljust(width)}  {ratio}  {s['type']:<8}  {s['example']}")

    if report.get("status_distribution"):
        print("\n-- *_status value distribution --")
        for k, dist in report["status_distribution"].items():
            pretty = {str(kk): vv for kk, vv in dist.items()}
            print(f"  {k}: {pretty}")

    if report.get("json_subkeys"):
        print("\n-- JSON sub-keys (parsed from string columns) --")
        for k, dist in report["json_subkeys"].items():
            print(f"  {k}: {dict(dist)}")


def count_rows(
    partition: str | None = None,
    table_name: str = EGOCENTRIC_DATASET_CLIPS_TABLE,
) -> int:
    """Exact row count using PostgREST count=exact (head request)."""
    q = supabase.table(table_name).select("path", count="exact").limit(1)
    if partition:
        q = q.eq("partition", partition)
    resp = q.execute()
    return resp.count or 0


# =============================================================================
# Script entry point: analyse both partitions
# =============================================================================

def main() -> None:
    print(f"target table: {EGOCENTRIC_DATASET_CLIPS_TABLE}")
    print(f"partitions  : {EGOCENTRIC_PARTITIONS}")

    for part in EGOCENTRIC_PARTITIONS:
        try:
            total = count_rows(partition=part)
            print(f"\n[{part}] total rows = {total}")
        except APIError as e:
            print(f"[{part}] count failed: {_postgrest_error_summary(e)}")

        try:
            report = analyze_partition(part, sample_size=500)
            print_partition_report(report)
        except APIError as e:
            print(f"[{part}] analyze failed: {_postgrest_error_summary(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
