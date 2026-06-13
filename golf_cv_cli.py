from __future__ import annotations

import argparse
import sys


ACTION_COMMANDS = {
    "annotate": "annotate",
    "train": "train",
    "generate-dataset": "generate-dataset",
    "generatedataset": "generate-dataset",
    "dataset": "generate-dataset",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="golf_cv",
        description="Golf CV tracker command launcher.",
        epilog=(
            "examples:\n"
            "  golf_cv\n"
            "  golf_cv --action annotate\n"
            "  golf_cv --action generate-dataset --dataset-config dataset_config.yaml\n"
            "  golf_cv --action train --config action_classifier_config.toml\n"
            "  golf_cv --import video data/raw_videos"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show this help.")
    parser.add_argument(
        "--action",
        metavar="{annotate,train,generate-dataset}",
        help=(
            "Run an action-classifier workflow: annotate, train, or "
            "generate-dataset."
        ),
    )
    parser.add_argument(
        "--import",
        dest="import_mode",
        choices=["video"],
        help="Import/run an external input through an offline workflow.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the main GUI application explicitly.",
    )

    args, forwarded = parser.parse_known_args(argv)
    selected = [bool(args.action), bool(args.import_mode), bool(args.gui)]
    if sum(selected) > 1:
        parser.error("Choose only one of --action, --import, or --gui.")
    if args.action and args.action not in ACTION_COMMANDS:
        parser.error(
            "Unknown --action value. Use annotate, train, or generate-dataset."
        )

    if args.help:
        if not any(selected) or args.gui:
            parser.print_help()
            return None
        forwarded = ["--help", *forwarded]

    if args.action:
        return _run_action(ACTION_COMMANDS[args.action], forwarded)

    if args.import_mode == "video":
        from video_test_runner import main as video_test_main

        return video_test_main(forwarded)

    from app_gui import main as app_main

    return app_main()


def _run_action(command, forwarded):
    if command == "annotate":
        if forwarded in (["--help"], ["-h"]):
            print("usage: golf_cv --action annotate")
            print()
            print("Open the interactive video interval annotation tool.")
            return None
        if forwarded:
            raise SystemExit("golf_cv --action annotate does not accept extra arguments.")
        from utils.annotate_intervals import main as annotate_main

        return annotate_main()

    if command == "train":
        from action.train import main as train_main

        return train_main(forwarded)

    if command == "generate-dataset":
        from action.generate_dataset import main as generate_dataset_main

        return generate_dataset_main(forwarded)

    raise SystemExit(f"Unknown action command: {command}")


if __name__ == "__main__":
    main()
