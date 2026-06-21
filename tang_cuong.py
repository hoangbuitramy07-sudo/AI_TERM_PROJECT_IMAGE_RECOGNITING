
import os
import cv2
import re
import random
import hashlib
import numpy as np
from PIL import Image
from unicodedata import normalize
from tqdm import tqdm
import albumentations as A


ORIGINAL_DATASET_DIR = r"D:\Downloads\food\TRAY FOOD AI"
AUGMENTED_DATASET_DIR = r"D:\Downloads\food\TRAY FOOD AI_augmented3"

IMG_SIZE = 128

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

TARGET_TRAIN_IMAGES = 800

random.seed(42)


SPLIT_RECORD = {} 


def clean_folder_name(text):
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'[\s]+', '_', text)
    return text


transform_pipeline = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),

    A.Rotate(
        limit=30,
        p=0.8,
        border_mode=cv2.BORDER_REFLECT
    ),

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=0,
        p=0.5,
        border_mode=cv2.BORDER_REFLECT
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),

    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    ),
])

base_transform = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE)
])


print("\n========== START ==========\n")

if not os.path.exists(ORIGINAL_DATASET_DIR):
    print(f"[ERROR] Không tìm thấy thư mục: {ORIGINAL_DATASET_DIR}")
    exit()

classes = [
    d for d in os.listdir(ORIGINAL_DATASET_DIR)
    if os.path.isdir(os.path.join(ORIGINAL_DATASET_DIR, d))
]

print(f"[FOUND] {len(classes)} classes\n")

for cls in classes:

    src_cls_dir = os.path.join(ORIGINAL_DATASET_DIR, cls)
    cls_clean = clean_folder_name(cls)

    all_files = os.listdir(src_cls_dir)

    img_paths = [
        os.path.join(src_cls_dir, f)
        for f in all_files
        if os.path.isfile(os.path.join(src_cls_dir, f))
        and f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        )
    ]

    if len(img_paths) == 0:
        continue

    random.shuffle(img_paths)

    total_imgs = len(img_paths)

    train_count = int(total_imgs * TRAIN_SPLIT)
    val_count = int(total_imgs * VAL_SPLIT)

    train_paths = img_paths[:train_count]
    val_paths = img_paths[train_count:train_count + val_count]
    test_paths = img_paths[train_count + val_count:]

    # GHI LẠI ĐỂ DÙNG CHO CHECK LEAKAGE / DISTRIBUTION
    SPLIT_RECORD[cls_clean] = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths
    }

    print("=" * 60)
    print(f"CLASS: {cls}")
    print(f"TOTAL : {total_imgs}")
    print(f"TRAIN : {len(train_paths)}")
    print(f"VAL   : {len(val_paths)}")
    print(f"TEST  : {len(test_paths)}")

    splits = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths
    }

    # TÍNH SỐ AUGMENT CẦN TẠO CHO MỖI ẢNH TRAIN
    if len(train_paths) > 0:
        variants_per_image = max(
            1,
            TARGET_TRAIN_IMAGES // len(train_paths) - 1
        )
    else:
        variants_per_image = 1

    print(
        f"Augment/Image = {variants_per_image}"
    )

    expected_train = len(train_paths) * (1 + variants_per_image)

    print(
        f"Expected Train Images ≈ {expected_train}"
    )

    for split_name, paths in splits.items():

        if len(paths) == 0:
            continue

        dst_cls_dir = os.path.join(
            AUGMENTED_DATASET_DIR,
            split_name,
            cls_clean
        )

        os.makedirs(dst_cls_dir, exist_ok=True)

        for img_path in tqdm(
            paths,
            desc=f"{cls_clean} -> {split_name}",
            leave=False
        ):

            try:
                pil_img = Image.open(img_path).convert("RGB")

                image = np.array(pil_img)

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )

                filename = os.path.basename(img_path)
                name, _ = os.path.splitext(filename)

                # LƯU ẢNH GỐC RESIZE
                resized = base_transform(
                    image=image
                )["image"]

                out_original = os.path.join(
                    dst_cls_dir,
                    f"{name}_orig.jpg"
                )

                Image.fromarray(
                    cv2.cvtColor(
                        resized,
                        cv2.COLOR_BGR2RGB
                    )
                ).save(
                    out_original,
                    quality=95
                )

                # AUGMENT CHỈ TRAIN
                if split_name == "train":

                    for i in range(
                        variants_per_image
                    ):

                        aug_img = transform_pipeline(
                            image=image
                        )["image"]

                        out_aug = os.path.join(
                            dst_cls_dir,
                            f"{name}_aug_{i+1}.jpg"
                        )

                        Image.fromarray(
                            cv2.cvtColor(
                                aug_img,
                                cv2.COLOR_BGR2RGB
                            )
                        ).save(
                            out_aug,
                            quality=95
                        )

            except Exception:
                pass

print("\n========================================")
print("HOÀN THÀNH AUGMENT")
print(f"OUTPUT: {AUGMENTED_DATASET_DIR}")
print("========================================")



def file_md5(path, block_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data_leakage(split_record=SPLIT_RECORD):
    print("\n========== CHECK DATA LEAKAGE ==========\n")

    # Gom hash theo từng split (toàn bộ class), set "path" để debug nếu cần
    hash_sets = {"train": {}, "val": {}, "test": {}}

    print("[1/2] Đang hash toàn bộ ảnh gốc...")
    for cls_clean, splits in tqdm(split_record.items(), desc="Hashing classes"):
        for split_name in ["train", "val", "test"]:
            for path in splits.get(split_name, []):
                try:
                    h = file_md5(path)
                    hash_sets[split_name][h] = path
                except Exception:
                    pass

    print("\n[2/2] So sánh chéo các cặp tập...\n")

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    results = {}

    for a, b in pairs:
        set_a = set(hash_sets[a].keys())
        set_b = set(hash_sets[b].keys())
        overlap = set_a & set_b
        results[f"{a}-{b}"] = len(overlap)

        print(f"{a.upper():5s} ∩ {b.upper():5s} : {len(overlap)} ảnh trùng "
              f"(kỳ vọng = 0)")

        # In ra vài ví dụ cụ thể nếu có leakage (để debug)
        if overlap:
            print("   --> Ví dụ file trùng:")
            for h in list(overlap)[:5]:
                print(f"       {hash_sets[a][h]}  <-->  {hash_sets[b][h]}")

    total_leak = sum(results.values())

    print("\n----------------------------------------")
    if total_leak == 0:
        print("[OK] Không phát hiện leakage giữa các tập (train/val/test).")
    else:
        print(f"[WARNING] Tổng số ảnh leakage phát hiện: {total_leak}")
    print("----------------------------------------\n")

    return results


def print_distribution(split_record=SPLIT_RECORD):
    print("\n========== PHÂN BỐ SỐ ẢNH GỐC THEO LỚP ==========\n")

    header = f"{'CLASS':35s} {'TRAIN':>8s} {'VAL':>8s} {'TEST':>8s} {'TOTAL':>8s}"
    print(header)
    print("-" * len(header))

    total_train = total_val = total_test = 0

    for cls_clean in sorted(split_record.keys()):
        splits = split_record[cls_clean]
        n_train = len(splits.get("train", []))
        n_val = len(splits.get("val", []))
        n_test = len(splits.get("test", []))
        n_total = n_train + n_val + n_test

        total_train += n_train
        total_val += n_val
        total_test += n_test

        print(f"{cls_clean:35s} {n_train:>8d} {n_val:>8d} {n_test:>8d} {n_total:>8d}")

    print("-" * len(header))
    grand_total = total_train + total_val + total_test
    print(f"{'TỔNG':35s} {total_train:>8d} {total_val:>8d} {total_test:>8d} {grand_total:>8d}")
    print()

    return {
        "train": total_train,
        "val": total_val,
        "test": total_test,
        "total": grand_total
    }


leakage_results = check_data_leakage()
distribution_results = print_distribution()