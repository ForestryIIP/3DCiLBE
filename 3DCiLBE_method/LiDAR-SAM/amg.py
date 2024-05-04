import cv2
import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    required=True,
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda"
)

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
)

amg_settings = parser.add_argument_group()

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
