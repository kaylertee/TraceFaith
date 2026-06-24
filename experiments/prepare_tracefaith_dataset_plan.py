from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.dataset_plan import DEFAULT_PLAN_PATH, create_dataset_plan
from aria.loaders import load_mixed_tracefaith_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the stable TraceFaith 500-per-dataset manifest.")
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / DEFAULT_PLAN_PATH)
    parser.add_argument("--per-dataset-limit", type=int, default=500)
    parser.add_argument("--scienceqa-split", default="validation")
    parser.add_argument("--ai2d-split", default=None)
    parser.add_argument("--mmmu-pro-split", default=None)
    parser.add_argument("--ai2d-local-jsonl", type=Path, default=None)
    parser.add_argument("--mmmu-pro-local-jsonl", type=Path, default=None)
    parser.add_argument("--mmmu-pro-include-non-science", action="store_true")
    parser.add_argument(
        "--scienceqa-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "scienceqa_images",
    )
    parser.add_argument(
        "--ai2d-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ai2d_images",
    )
    parser.add_argument(
        "--mmmu-pro-image-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "mmmu_pro_images",
    )
    parser.add_argument("--require-explanation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scienceqa-diagram-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, dataset_config = load_mixed_tracefaith_datasets(
        scienceqa_count=args.per_dataset_limit,
        ai2d_count=args.per_dataset_limit,
        mmmu_pro_count=args.per_dataset_limit,
        scienceqa_split=args.scienceqa_split,
        ai2d_split=args.ai2d_split,
        mmmu_pro_split=args.mmmu_pro_split,
        scienceqa_image_dir=args.scienceqa_image_dir / args.scienceqa_split,
        ai2d_image_dir=args.ai2d_image_dir / (args.ai2d_split or "auto"),
        mmmu_pro_image_dir=args.mmmu_pro_image_dir / (args.mmmu_pro_split or "auto"),
        ai2d_local_jsonl_path=args.ai2d_local_jsonl,
        mmmu_pro_local_jsonl_path=args.mmmu_pro_local_jsonl,
        mmmu_pro_science_subjects_only=not args.mmmu_pro_include_non_science,
        require_explanation=args.require_explanation,
        diagram_only=args.scienceqa_diagram_only,
    )
    planned = create_dataset_plan(
        examples=examples,
        output_path=args.output_jsonl,
        per_dataset_limit=args.per_dataset_limit,
    )
    counts = {
        "scienceqa": sum(example.dataset == "scienceqa" for example in planned),
        "ai2d": sum(example.dataset == "ai2d" for example in planned),
        "mmmu_pro": sum(example.dataset == "mmmu_pro" for example in planned),
    }
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "counts": counts, **dataset_config}, indent=2))


if __name__ == "__main__":
    main()
