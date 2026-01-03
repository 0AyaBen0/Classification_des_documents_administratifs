# src/pdf_to_image.py

import os
from pdf2image import convert_from_path


def convert_pdf_to_images(
    pdf_path,
    output_base_dir,
    dpi=300
):
    """
    Convert a PDF into images (one image per page).

    Args:
        pdf_path (str): path to the input PDF
        output_base_dir (str): base directory where images will be saved
        dpi (int): resolution for conversion
    """

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    output_dir = os.path.join(output_base_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        pages = convert_from_path(pdf_path, dpi=dpi)

        for i, page in enumerate(pages):
            image_path = os.path.join(
                output_dir, f"page_{i+1}.jpg"
            )
            page.save(image_path, "JPEG")

        print(f"[OK] {pdf_name}: {len(pages)} pages converted")

    except Exception as e:
        print(f"[ERROR] Failed to convert {pdf_path}")
        print(e)


def process_dataset(
    raw_pdfs_dir="data/raw_pdfs",
    output_images_dir="data/images"
):
    """
    Loop over all classes and PDFs in the dataset.
    """

    for class_name in os.listdir(raw_pdfs_dir):
        class_path = os.path.join(raw_pdfs_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(output_images_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for file in os.listdir(class_path):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(class_path, file)
                convert_pdf_to_images(pdf_path, output_class_dir)


if __name__ == "__main__":
    process_dataset()
