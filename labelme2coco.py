#!/usr/bin/env python
import tqdm
import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
# import cv2
import imgviz
import numpy as np

import labelme
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def filter_jpeg_png(img_file):
    if img_file.split('.')[1].lower() in ('jpeg', 'png'):
        return True
    return False


def main(directory):
    isSucess = True
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument('--name',help='input type of data(ARCH|STR)',default=None)

    parser.add_argument("--labels", help="labels file", required=False,
                        default=osp.join(osp.dirname(__file__), 'labels.txt'))
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true", default=True
    )
    args = parser.parse_args()

    if args.name:
        if (args.name).lower() == 'str':
            args.labels = osp.join(osp.dirname(__file__),'labels_str.txt')
        else:
            args.labels = osp.join(osp.dirname(__file__),'labels_arch.txt')
    else:
        args.labels = osp.join(osp.dirname(__file__), 'labels.txt')

    # if osp.exists(args.output_dir):
    # print("Output directory already exists:", args.output_dir)
    # sys.exit(1)
    args.output_dir = osp.join(args.output_dir, directory)
    # if osp.exists(args.output_dir):
    #     print(f"Directory {args.output_dir} already existed, removing and recreating it...")
    #     os.remove(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )
    args.input_dir = os.path.join(args.input_dir, directory)
    out_ann_file = osp.join(args.output_dir, f"instances_{directory}2017.json")

    # process image files

    # jpeg_png_images = list(filter(filter_jpeg_png,image_files))
    # print(f"jpeg_png_images: {jpeg_png_images}")
    label_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith('.json')]

    # Bug: loss of images after conversion is done; check for image with extension from jpg,jpeg,png
    # if present at input location, add it to the list of images and repeat for all label files

    image_files = []
    for label_file in label_files:
        basename = osp.splitext(osp.basename(label_file))[0]
        for extension in ('jpg','jpeg','png'):
            out_img = f"{basename}.{extension}"
            if osp.exists(osp.join(args.input_dir,out_img)):
                image_files.append(osp.join(args.input_dir,out_img))
                break

    print(f"directory: {directory} Image files: {len(image_files)} label_files:{len(label_files)}")
    for image_id, filename in tqdm.tqdm(enumerate(label_files), desc='processing labels...'):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]

        has_corr_image_file = False
        for extension in ('jpg', 'jpeg', 'png'):
            out_img = f"{base}.{extension}"
            if osp.exists(osp.join(args.input_dir, out_img)):
                has_corr_image_file = True
                break
        # for img_file in image_files:
        #     if img_file.startswith(base):
        #         out_img = img_file
        #         break
        if has_corr_image_file:
            # print(out_img)

            out_img_file = osp.join(args.output_dir, out_img)
            # shutil.copy(filename,args.output_dir)
            try:
                img = labelme.utils.img_data_to_arr(label_file.imageData)
                imgviz.io.imsave(out_img_file, img)
                # cv2.imwrite(out_img_file,img)
                # shutil.copy(os.path.join(args.input_dir,out_img),out_img_file)

            except Exception as E:
                print("Exception to copy ", E)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            if not args.noviz:
                viz = img
                if masks:
                    labels, captions, masks = zip(
                        *[
                            (class_name_to_id[cnm], cnm, msk)
                            for (cnm, gid), msk in masks.items()
                            if cnm in class_name_to_id
                        ]
                    )
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                out_viz_file = osp.join(
                    args.output_dir, "Visualization", base + ".jpg"
                )
                imgviz.io.imsave(out_viz_file, viz)
        else:
            print(f"Could not find image file for label: {filename}")
            isSucess = False
    with open(out_ann_file, "w") as f:
        json.dump(data, f)
    return isSucess, args.output_dir


if __name__ == "__main__":
    output_dirs = list()
    for directory in ('train', 'test', 'val'):
        isSuccess, output_dir = main(directory)
        if isSuccess:
            output_dirs.append(output_dir)
    annotation_folder = osp.join(output_dirs[0], '..', 'annotations')
    os.makedirs(annotation_folder)
    for directory in output_dirs:
        shutil.move(os.path.join(directory, f"instances_{osp.basename(directory)}2017.json"), annotation_folder)
        os.rename(directory, os.path.join(os.path.dirname(directory), f"{os.path.basename(directory)}2017"))
