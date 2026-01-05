import cv2
import os


def preprocess_image(
    image_path,
    output_path,
    target_size=(1024, 1024)
):
    """
    Apply basic preprocessing to an image:
    - resize
    - grayscale
    - light denoising

    Args:
        image_path (str): path to input image
        output_path (str): path to save preprocessed image
        target_size (tuple): resize dimensions
    """

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # Resize
    image_resized = cv2.resize(image, target_size)

    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Light denoising
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, denoised)


def preprocess_dataset(
    input_dir="data/images",
    output_dir="data/preprocessed_images"
):
    """
    Apply preprocessing to all images in the dataset.
    """

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for pdf_folder in os.listdir(class_path):
            pdf_path = os.path.join(class_path, pdf_folder)

            if not os.path.isdir(pdf_path):
                continue

            output_pdf_dir = os.path.join(
                output_dir, class_name, pdf_folder
            )
            os.makedirs(output_pdf_dir, exist_ok=True)

            for file in os.listdir(pdf_path):
                if file.lower().endswith(".jpg"):
                    input_image_path = os.path.join(pdf_path, file)
                    output_image_path = os.path.join(output_pdf_dir, file)

                    preprocess_image(
                        input_image_path,
                        output_image_path
                    )

    print("[OK] Image preprocessing completed")


if __name__ == "__main__":
    preprocess_dataset()
