import argparse
import os
import random

TRAIN_SIZE = 6000
VAL_SIZE   = 1000
TEST_SIZE  = 1000
SEED       = 42

def main(images_dir: str, output_dir: str) -> None:
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg")
    ])

    total = len(all_images)
    print(f"Total gambar ditemukan: {total}")

    if total < TRAIN_SIZE + VAL_SIZE + TEST_SIZE:
        raise ValueError(
            f"Gambar tidak cukup! Butuh {TRAIN_SIZE+VAL_SIZE+TEST_SIZE}, "
            f"tapi hanya ada {total}."
        )

    random.seed(SEED)
    random.shuffle(all_images)

    train = all_images[:TRAIN_SIZE]
    val   = all_images[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test  = all_images[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "Flickr_8k.trainImages.txt": train,
        "Flickr_8k.devImages.txt":   val,
        "Flickr_8k.testImages.txt":  test,
    }

    for filename, image_list in splits.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write("\n".join(image_list))
        print(f"Tersimpan: {path} ({len(image_list)} gambar)")

    print("\nSplit selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/Images")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    main(args.images_dir, args.output_dir)
