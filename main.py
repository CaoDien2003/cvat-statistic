import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cvat_stats.parser import parse_xml, to_image_df, read_xml_date
from cvat_stats.stats import summary_stats, label_stats, group_stats, user_progress, sorted_labels
from cvat_stats.history import (
    save_snapshot,
    load_previous_snapshot,
    compute_delta,
    delete_snapshot,
    list_snapshots,
)
from cvat_stats.exporter import export_excel, export_project_csv, append_daily_worker_row


def resolve_run_date(xml_path: Path, cli_date: Optional[str]) -> date:
    if cli_date:
        return date.fromisoformat(cli_date)
    xml_date = read_xml_date(xml_path)
    if xml_date:
        return xml_date
    return date.today()


def process_single_xml(
    xml_path: Path,
    run_date: date,
    baseline_date: Optional[date],
    output_dir: Path,
) -> Path:
    print(f"\n[Processing] {xml_path.name}")

    parsed = parse_xml(xml_path)
    labels = parsed.meta.labels
    label_types = parsed.meta.label_types
    project_name = parsed.meta.project_name

    print(f"  Project : {project_name}")
    print(f"  Date    : {run_date}")
    print(f"  Images  : {len(parsed.images)}")
    print(f"  Labels  : {len(labels)}")

    img_df = to_image_df(parsed)
    labeled_count = int(img_df["is_labeled"].sum())
    total_bboxes = sum(int(img_df[f"bbox_{lbl}"].sum()) for lbl in labels if f"bbox_{lbl}" in img_df.columns)
    print(f"  Labeled : {labeled_count} / {len(img_df)}")
    print(f"  BBoxes  : {total_bboxes} total across all classes")

    summary_df = summary_stats(parsed, img_df)
    label_df = label_stats(img_df, labels, label_types, labeled_count)
    group_df = group_stats(img_df, labels, label_types)
    user_df = user_progress(img_df, labels, label_types)

    prev_snap = load_previous_snapshot(project_name, before_date=run_date, baseline_date=baseline_date)
    if prev_snap:
        print(f"  History : comparing vs snapshot {prev_snap['date']}")
    else:
        print("  History : no previous snapshot found — first run")

    delta_df = compute_delta(img_df, prev_snap, labels)
    save_snapshot(project_name, img_df, run_date)

    safe_name = project_name.replace(" ", "_").replace("/", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = output_dir / f"summary_{run_date.isoformat()}_{safe_name}.xlsx"
    export_excel(excel_path, summary_df, label_df, group_df, img_df, user_df, delta_df, labels)
    print(f"  Excel   : {excel_path}")

    project_csv = output_dir / f"{safe_name}.csv"
    export_project_csv(project_csv, img_df, sorted_labels(labels, label_types))
    print(f"  CSV     : {project_csv}")

    worker_csv = output_dir / f"daily_worker_{safe_name}.csv"
    append_daily_worker_row(worker_csv, run_date, project_name, user_df)
    print(f"  Worker  : {worker_csv}")

    return excel_path


def cmd_list_history(args):
    snapshots = list_snapshots()
    if not snapshots:
        print("No history snapshots found.")
        return
    print(f"{'Date':<12}  Project")
    print("-" * 50)
    for s in snapshots:
        print(f"{s['date']:<12}  {s['project_name']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CVAT annotation statistics — produces Excel + CSV reports with daily delta tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("xml", nargs="*", metavar="XML_FILE", help="CVAT annotation XML file(s)")
    parser.add_argument("-d", "--date", metavar="YYYY-MM-DD")
    parser.add_argument("--history", metavar="YYYY-MM-DD")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--list-history", action="store_true")
    parser.add_argument("-o", "--output", metavar="DIR", default="output")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_history:
        cmd_list_history(args)
        return

    if not args.xml:
        parser.print_help()
        sys.exit(1)

    xml_paths = [Path(p) for p in args.xml]
    missing = [p for p in xml_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] File not found: {p}")
        sys.exit(1)

    baseline_date = date.fromisoformat(args.history) if args.history else None
    output_dir = Path(args.output)

    results = []
    for xml_path in xml_paths:
        run_date = resolve_run_date(xml_path, args.date)

        if args.revert:
            parsed_meta = parse_xml(xml_path).meta
            removed = delete_snapshot(parsed_meta.project_name, run_date)
            if removed:
                print(f"[Revert] Deleted snapshot {run_date} for {parsed_meta.project_name}")
            else:
                print(f"[Revert] No snapshot found for {run_date} / {parsed_meta.project_name} — proceeding fresh")

        out = process_single_xml(xml_path, run_date, baseline_date, output_dir)
        results.append(out)

    print(f"\nDone. {len(results)} report(s) written:")
    for r in results:
        print(f"  -> {r}")


if __name__ == "__main__":
    main()
