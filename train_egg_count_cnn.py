import argparse
import json
import os

import tensorflow as tf


IMG_SIZE = (192, 192)
BATCH_SIZE = 16
SEED = 42


def numeric_class_names(data_dir):
    class_names = []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and name.isdigit():
            class_names.append(name)

    if len(class_names) < 2:
        raise ValueError(
            "Dataset cần ít nhất 2 thư mục class dạng số, ví dụ: 1, 2, 3."
        )

    return sorted(class_names, key=lambda value: int(value))


def build_model(num_classes):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN đếm số trứng cho món thịt kho trứng."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "egg_count_dataset"),
        help="Thư mục dataset. Mỗi class là một thư mục số: 1, 2, 3, ...",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "egg_count_cnn.keras"),
    )
    parser.add_argument(
        "--labels-output",
        default=os.path.join(os.path.dirname(__file__), "egg_count_labels.json"),
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data)
    class_names = numeric_class_names(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(512).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output)
    with open(args.labels_output, "w", encoding="utf-8") as f:
        json.dump([int(name) for name in class_names], f, ensure_ascii=False, indent=2)

    print(f"Saved model: {args.output}")
    print(f"Saved labels: {args.labels_output}")
    print(f"Classes: {', '.join(class_names)}")


if __name__ == "__main__":
    main()
