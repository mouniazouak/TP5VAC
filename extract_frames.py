import argparse
import csv
from pathlib import Path

import numpy as np
import av
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from UCF101 videos"
    )
    parser.add_argument(
        "input_root",
        help="Folder containing the UCF101 .avi videos",
    )
    parser.add_argument(
        "--output-root",
        default="frames",
        help="Folder used to save the extracted numpy arrays",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.csv",
        help="CSV file containing the video metadata",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames sampled from each video",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=80,
        help="Size of the extracted frames",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Folder containing trainlist01.txt and testlist01.txt",
    )
    return parser.parse_args()


def get_video_group(video_path: Path) -> str:
    """Return the group identifier used for the train/validation split."""
    return video_path.stem.rsplit("_", 1)[0]


def load_official_splits(annotation_root: Path | None):
    """Load UCF101 train/test lists when the official annotations are present."""
    if annotation_root is None or not annotation_root.exists():
        return set(), set()

    train_files = sorted(annotation_root.rglob("trainlist01.txt"))
    test_files = sorted(annotation_root.rglob("testlist01.txt"))

    def read_list(paths):
        values = set()
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip().split()[0] if line.strip() else ""
                if value:
                    values.add(value.replace("\\", "/").lower())
        return values

    return read_list(train_files), read_list(test_files)


def get_video_split(
    video_path: Path,
    input_root: Path,
    validation_groups: dict[str, set[str]],
    official_train: set[str],
    official_test: set[str],
) -> str:
    """Use official lists when available, otherwise split by video group."""
    relative_path = video_path.relative_to(input_root).as_posix().lower()
    relative_name = video_path.name.lower()

    if relative_path in official_test or relative_name in official_test:
        return "test"
    if relative_path in official_train or relative_name in official_train:
        class_name = video_path.parent.name
        group = get_video_group(video_path)
        if group in validation_groups.get(class_name, set()):
            return "val"
        return "train"

    class_name = video_path.parent.name
    group = get_video_group(video_path)
    if group in validation_groups.get(class_name, set()):
        return "val"
    return "train"


def build_validation_groups(
    video_paths: list[Path],
    input_root: Path,
    official_train: set[str],
    official_test: set[str],
) -> dict[str, set[str]]:
    """Reserve validation groups independently inside every action class."""
    groups_by_class = {}

    for video_path in video_paths:
        relative_path = video_path.relative_to(input_root).as_posix().lower()
        relative_name = video_path.name.lower()
        is_test = relative_path in official_test or relative_name in official_test
        if official_train and not (
            relative_path in official_train or relative_name in official_train
        ):
            continue
        if is_test:
            continue
        class_name = video_path.parent.name
        groups_by_class.setdefault(class_name, set()).add(
            get_video_group(video_path)
        )

    validation_groups = {}
    for class_name, groups in groups_by_class.items():
        ordered_groups = sorted(groups)
        number_of_validation_groups = max(1, int(0.2 * len(ordered_groups)))
        validation_groups[class_name] = set(
            ordered_groups[:number_of_validation_groups]
        )
    return validation_groups


def extract_video_frames(
    video_path: Path,
    output_path: Path,
    number_of_frames: int,
    frame_size: int,
) -> bool:
    container = av.open(str(video_path))
    frames = []

    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())

    container.close()

    if not frames:
        print(f"Video vide ignorée : {video_path}")
        return False

    frame_indices = np.linspace(
        0,
        len(frames) - 1,
        number_of_frames,
        dtype=int,
    )

    extracted_frames = []

    for frame_index in frame_indices:
        frame = Image.fromarray(frames[frame_index])
        frame = frame.convert("RGB")
        frame = frame.resize((frame_size, frame_size))
        extracted_frames.append(np.asarray(frame))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        output_path,
        np.stack(extracted_frames).astype(np.uint8),
    )
    return True


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    annotation_root = Path(args.annotations) if args.annotations else None

    if not input_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {input_root}"
        )

    video_paths = sorted(input_root.rglob("*.avi"))

    if not video_paths:
        raise RuntimeError(
            "Aucune vidéo .avi n'a été trouvée dans le dossier."
        )

    class_names = sorted(
        {video_path.parent.name for video_path in video_paths}
    )
    official_train, official_test = load_official_splits(annotation_root)
    validation_groups = build_validation_groups(
        video_paths,
        input_root,
        official_train,
        official_test,
    )

    if not official_test:
        print(
            "Attention : aucune liste test officielle n'a été trouvée. "
            "Le manifeste contiendra train/val uniquement."
        )

    rows = []

    for video_number, video_path in enumerate(video_paths, start=1):
        relative_path = video_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".npy")

        if output_path.exists():
            try:
                saved = np.load(output_path, mmap_mode="r").shape[0] == args.frames
            except (OSError, ValueError):
                saved = False

            if not saved:
                saved = extract_video_frames(
                    video_path=video_path,
                    output_path=output_path,
                    number_of_frames=args.frames,
                    frame_size=args.size,
                )
        else:
            saved = extract_video_frames(
                video_path=video_path,
                output_path=output_path,
                number_of_frames=args.frames,
                frame_size=args.size,
            )

        if not saved:
            continue

        rows.append(
            {
                "frames": str(output_path),
                "label": class_names.index(video_path.parent.name),
                "class_name": video_path.parent.name,
                "group_id": get_video_group(video_path),
                "split": get_video_split(
                    video_path,
                    input_root,
                    validation_groups,
                    official_train,
                    official_test,
                ),
                "video_id": video_path.stem,
            }
        )

        print(
            f"Vidéo {video_number}/{len(video_paths)} traitée : "
            f"{video_path.name}"
        )

    if not rows:
        raise RuntimeError(
            "Aucune frame n'a pu être extraite."
        )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extraction terminée : {len(rows)} vidéos")
    print(f"Manifest enregistré dans : {manifest_path}")


if __name__ == "__main__":
    main()
