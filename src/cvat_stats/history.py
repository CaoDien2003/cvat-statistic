import json
import shutil
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

HISTORY_DIR = Path(__file__).parent.parent.parent / "history"
REGISTRY_PATH = HISTORY_DIR / "registry.json"


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def _project_dir(safe_name: str) -> Path:
    return HISTORY_DIR / safe_name


def _project_log_path(safe_name: str) -> Path:
    return _project_dir(safe_name) / "log.json"


def _snapshot_path(project_name: str, run_date: date) -> Path:
    return _project_dir(_safe(project_name)) / f"{run_date.isoformat()}.json"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(registry: dict) -> None:
    HISTORY_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def register_project(name: str, cvat_id: str = "") -> str:
    key = _safe(name)
    reg = load_registry()
    if key not in reg:
        reg[key] = {"name": name, "cvat_id": cvat_id, "created": date.today().isoformat()}
        _save_registry(reg)
    _project_dir(key).mkdir(parents=True, exist_ok=True)
    return key


def delete_project(name: str) -> bool:
    key = _safe(name)
    reg = load_registry()
    removed = False
    if key in reg:
        del reg[key]
        _save_registry(reg)
        removed = True
    proj_dir = _project_dir(key)
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
        removed = True
    old_log = HISTORY_DIR / f"project_{key}.json"
    if old_log.exists():
        old_log.unlink()
        removed = True
    return removed


def rename_project(old_name: str, new_name: str) -> bool:
    old_key = _safe(old_name)
    new_key = _safe(new_name)
    if old_key == new_key:
        return False
    reg = load_registry()
    if old_key not in reg or new_key in reg:
        return False

    old_dir = _project_dir(old_key)
    new_dir = _project_dir(new_key)

    if old_dir.exists():
        old_dir.rename(new_dir)

    log_path = new_dir / "log.json"
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            log_data["project_name"] = new_name
            log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    if new_dir.exists():
        for snap_path in new_dir.glob("*.json"):
            if snap_path.name == "log.json":
                continue
            try:
                snap = json.loads(snap_path.read_text(encoding="utf-8"))
                if "project_name" in snap:
                    snap["project_name"] = new_name
                    snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            except Exception:
                pass

    entry = reg.pop(old_key)
    entry["name"] = new_name
    reg[new_key] = entry
    _save_registry(reg)
    return True


def migrate_flat_to_subdir() -> int:
    if not HISTORY_DIR.exists():
        return 0
    moved = 0

    for path in list(HISTORY_DIR.glob("project_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            proj_name = data.get("project_name", "")
            if not proj_name:
                continue
            safe = _safe(proj_name)
            dest_dir = _project_dir(safe)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "log.json"
            if not dest.exists():
                dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            path.unlink()
            moved += 1
        except Exception:
            pass

    for path in list(HISTORY_DIR.glob("*.json")):
        if path.name == "registry.json" or path.name.startswith("project_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "images" not in data:
                continue
            proj_name = data.get("project_name", "")
            snap_date = data.get("date", "")
            if not proj_name or not snap_date:
                continue
            safe = _safe(proj_name)
            dest_dir = _project_dir(safe)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{snap_date}.json"
            if not dest.exists():
                dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            path.unlink()
            moved += 1
        except Exception:
            pass

    return moved


def save_project_log_entry(
    project_name: str,
    run_date: date,
    total_images: int,
    labeled: int,
    by_user: Dict[str, int],
    cvat_id: str = "",
) -> Path:
    key = register_project(project_name, cvat_id)
    log_path = _project_log_path(key)

    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        log_data = {"project_name": project_name, "cvat_id": cvat_id, "entries": []}

    date_str = run_date.isoformat()
    log_data["entries"] = [e for e in log_data["entries"] if e["date"] != date_str]
    log_data["entries"].append({
        "date": date_str,
        "total_images": total_images,
        "labeled": labeled,
        "by_user": {u: v for u, v in by_user.items() if u},
    })
    log_data["entries"].sort(key=lambda e: e["date"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def load_project_log(project_name: str) -> dict:
    log_path = _project_log_path(_safe(project_name))
    if log_path.exists():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"project_name": project_name, "cvat_id": "", "entries": []}


def delete_project_log_entry(project_name: str, run_date: date) -> bool:
    log_path = _project_log_path(_safe(project_name))
    if not log_path.exists():
        return False
    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    before = len(log_data["entries"])
    log_data["entries"] = [e for e in log_data["entries"] if e["date"] != run_date.isoformat()]
    if len(log_data["entries"]) < before:
        log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    return False


def build_progress_table(log: dict) -> pd.DataFrame:
    entries = sorted(log.get("entries", []), key=lambda e: e["date"])
    if not entries:
        return pd.DataFrame()

    all_users: set = set()
    for e in entries:
        all_users |= set(e.get("by_user", {}).keys())
    users = sorted(all_users)

    date_cols: List[str] = []
    user_deltas: Dict[str, List[int]] = {u: [] for u in users}
    total_deltas: List[int] = []

    for i, entry in enumerate(entries):
        y, m, d_ = entry["date"].split("-")
        col = f"{m}/{d_}/{y[2:]}"
        date_cols.append(col)

        prev = entries[i - 1] if i > 0 else None
        day_total = 0
        for u in users:
            curr = entry.get("by_user", {}).get(u, 0)
            prev_val = prev.get("by_user", {}).get(u, 0) if prev else 0
            delta = max(0, curr - prev_val)
            user_deltas[u].append(delta)
            day_total += delta
        total_deltas.append(day_total)

    latest = entries[-1]
    rows = []
    for u in users:
        row: dict = {"user": u}
        for col, val in zip(date_cols, user_deltas[u]):
            row[col] = val
        row["total"] = latest.get("by_user", {}).get(u, 0)
        rows.append(row)

    total_row: dict = {"user": "TOTAL"}
    for col, val in zip(date_cols, total_deltas):
        total_row[col] = val
    total_row["total"] = latest.get("labeled", 0)
    rows.append(total_row)

    return pd.DataFrame(rows).set_index("user")


def delete_snapshot(project_name: str, run_date: date) -> bool:
    path = _snapshot_path(project_name, run_date)
    if path.exists():
        path.unlink()
        return True
    old = HISTORY_DIR / f"{run_date.isoformat()}_{_safe(project_name)}.json"
    if old.exists():
        old.unlink()
        return True
    return False


def list_snapshots(project_name: Optional[str] = None) -> List[dict]:
    if not HISTORY_DIR.exists():
        return []

    results = []

    dirs_to_scan = (
        [_project_dir(_safe(project_name))] if project_name
        else [d for d in HISTORY_DIR.iterdir() if d.is_dir()]
    )
    for proj_dir in dirs_to_scan:
        if not proj_dir.exists():
            continue
        for path in sorted(proj_dir.glob("*.json"), reverse=True):
            if path.name == "log.json":
                continue
            try:
                snap = json.loads(path.read_text(encoding="utf-8"))
                if "images" in snap:
                    results.append({
                        "date": snap["date"],
                        "project_name": snap["project_name"],
                        "path": path,
                    })
            except Exception:
                pass

    pattern = f"*_{_safe(project_name)}.json" if project_name else "*.json"
    for path in sorted(HISTORY_DIR.glob(pattern), reverse=True):
        if path.name == "registry.json" or path.name.startswith("project_"):
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
            if "images" in snap:
                results.append({
                    "date": snap["date"],
                    "project_name": snap["project_name"],
                    "path": path,
                })
        except Exception:
            pass

    return sorted(results, key=lambda x: x["date"], reverse=True)


def load_snapshot_by_date(project_name: str, snap_date: date) -> Optional[dict]:
    path = _snapshot_path(project_name, snap_date)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    old = HISTORY_DIR / f"{snap_date.isoformat()}_{_safe(project_name)}.json"
    if old.exists():
        return json.loads(old.read_text(encoding="utf-8"))
    return None


def save_snapshot(project_name: str, img_df: pd.DataFrame, run_date: Optional[date] = None) -> Path:
    run_date = run_date or date.today()
    snapshot = {
        "date": run_date.isoformat(),
        "project_name": project_name,
        "images": img_df.set_index("img_name").to_dict(orient="index"),
    }
    path = _snapshot_path(project_name, run_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path


def load_previous_snapshot(
    project_name: str,
    before_date: Optional[date] = None,
    baseline_date: Optional[date] = None,
) -> Optional[dict]:
    if baseline_date is not None:
        return load_snapshot_by_date(project_name, baseline_date)

    before_date = before_date or date.today()
    safe = _safe(project_name)

    candidates: List[Path] = []
    proj_dir = _project_dir(safe)
    if proj_dir.exists():
        candidates.extend(sorted(proj_dir.glob("*.json")))
    candidates.extend(sorted(HISTORY_DIR.glob(f"*_{safe}.json")))

    for path in reversed(candidates):
        if path.name == "log.json":
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
            if "images" not in snap:
                continue
            snap_date = date.fromisoformat(snap["date"])
            if snap_date < before_date:
                return snap
        except Exception:
            pass
    return None


def compute_delta(current_df: pd.DataFrame, previous_snap: Optional[dict], labels: List[str]) -> pd.DataFrame:
    if previous_snap is None:
        rows = []
        for user, grp in current_df.groupby("assignee"):
            row = {
                "assignee": user if user else "(unassigned)",
                "prev_labeled": 0,
                "curr_labeled": int(grp["is_labeled"].sum()),
                "delta_labeled": int(grp["is_labeled"].sum()),
                "note": "first run — no previous snapshot",
            }
            for lbl in labels:
                row[f"delta_{lbl}"] = int(grp[lbl].sum()) if lbl in grp.columns else 0
            rows.append(row)
        return pd.DataFrame(rows)

    prev_images = previous_snap.get("images", {})
    prev_user: Dict[str, dict] = {}
    for img_name, data in prev_images.items():
        user = data.get("assignee", "")
        if user not in prev_user:
            prev_user[user] = {"labeled": 0, **{lbl: 0 for lbl in labels}}
        prev_user[user]["labeled"] += int(data.get("is_labeled", 0))
        for lbl in labels:
            prev_user[user][lbl] += int(data.get(lbl, 0))

    rows = []
    for user, grp in current_df.groupby("assignee"):
        key = user if user else ""
        prev = prev_user.get(key, {"labeled": 0, **{lbl: 0 for lbl in labels}})
        curr_labeled = int(grp["is_labeled"].sum())
        row = {
            "assignee": user if user else "(unassigned)",
            "prev_date": previous_snap["date"],
            "prev_labeled": prev["labeled"],
            "curr_labeled": curr_labeled,
            "delta_labeled": curr_labeled - prev["labeled"],
        }
        for lbl in labels:
            curr_lbl = int(grp[lbl].sum()) if lbl in grp.columns else 0
            row[f"delta_{lbl}"] = curr_lbl - prev.get(lbl, 0)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("assignee")
