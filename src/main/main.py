import argparse
import os
import pathlib
import sys

import openai
import requests
from PIL import Image
from fitz import fitz
from pytesseract import pytesseract

DEFAULT_ATTRIBUTE_LIST = "attributes"
DEFAULT_LLM_MODEL = "gpt-4"
DEFAULT_WORKING_DIR = ".pdfextract"
DEFAULT_INSTRUCTIONS = "instructions"
DEFAULT_TESSERACT_COMMAND = r'/opt/homebrew/bin/tesseract'


def download_pdf(url: str, save_to: str) -> str:
    """
    Downloads a PDF file from a specified URL and saves it to a designated directory.

    Args:
        url (str): The URL from where the PDF file should be downloaded. The URL must point to a .pdf file,
                   otherwise, a ValueError is raised.
        save_to (str): The directory path where the PDF file will be saved. This directory must exist,
                       otherwise, a FileNotFoundError is raised. If the directory is not specified, the
                       current working directory is used.

    Returns:
        str: The full file path of the downloaded PDF if successful, None if the download fails.
             A successful download returns the file path, while failures due to non-200 HTTP status
             codes will raise an HTTPError with the reason for the failure.

    Raises:
        FileNotFoundError: Raised if the 'save_to' directory does not exist.
        ValueError: Raised if the URL does not end in ".pdf".
        requests.exceptions.HTTPError: Raised if the download fails due to server response issues.

    Note:
        The function checks if the response from the server is successful (HTTP status code 200) before
        attempting to save the file. It handles writing the file in binary mode to preserve the PDF format.
    """

    if not os.path.exists(save_to):
        raise FileNotFoundError(f'Working directory {save_to} not exist')

    if pathlib.Path(url).suffix.lower() != ".pdf":
        raise ValueError(f"{url} may not point to a PDF file (not with a .pdf extension)")

    # Request URL and get response object
    response = requests.get(url, stream=True)

    # isolate PDF filename from URL
    pdf_file_name = os.path.join(save_to, os.path.basename(url))

    if response.status_code == 200:
        # Save in current working directory
        filepath = os.path.join(os.getcwd(), pdf_file_name)
        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            return pdf_file_name
    else:
        raise requests.exceptions.HTTPError(f'Failed to download {url} ({response.status_code})')


def pdf2text(pdf_file: str, working_dir: str) -> str:
    """
    Extracts text from a PDF file, including text within images, using OCR (Optical Character Recognition).
    Saves the extracted text to a text file in the specified working directory.

    Args:
        pdf_file (str): The file path to the PDF from which text is to be extracted.
        working_dir (str): The directory where the text file and any intermediate images will be saved.

    Returns:
        str: The file path to the created text file containing all extracted text.

    Requires:
        - PyMuPDF (fitz): For handling PDF files and images within them.
        - PIL (Image): For opening and manipulating images.
        - pytesseract: For performing OCR on images.

    Note:
        The function assumes the availability of the pytesseract OCR tool and its proper configuration.
        Images are saved temporarily and are used for OCR. Consider cleaning up these images if space is a concern.

    Example:
        >>> extracted_file = pdf2text('sample.pdf', '/path/to/working/directory')
        >>> print('Extracted text saved to:', extracted_file)

    Raises:
        Various exceptions can be raised by the libraries used (fitz, PIL, pytesseract) if files
        are not found, if there are issues with file formats, or if OCR fails on image data.
    """

    # Create an output file name
    file_stem = pathlib.Path(pdf_file).stem
    txt_file = os.path.join(working_dir, file_stem + ".txt")

    # Open the PDF file using PyMuPDF
    doc = fitz.open(pdf_file)
    text = ''

    # Iterate through each page in the PDF document
    for p, page in enumerate(doc):
        # Extract text from the current page
        text += page.get_text()

        # Retrieve all images from the current page
        images = page.get_images()
        for i, img in enumerate(images):
            xref = img[0]  # image reference number in the PDF
            pix = fitz.Pixmap(doc, xref)  # Create a pixmap object from the image reference

            # Check if the image is in CMYK color space and convert to RGB if necessary
            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix = pix1
            elif pix.alpha == 1:
                pix1 = fitz.Pixmap(pix, 0)
                pix = pix1

            # # I was trying to convert pix directly to PIL Image without saving to disk.  No success.
            # # 'ValueError: not enough image data'
            # mode = "RGB" if pix.n == 3 else "RGBA"
            # img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            # Save the image to a PNG file in the working directory
            image_file = f"{working_dir}/{file_stem}_p{p}_i{i}.png"
            pix.save(image_file)

            # Open the saved image for OCR
            img = Image.open(image_file)

            # Use pytesseract to perform OCR on the image and append the text to the output
            extracted_text = pytesseract.image_to_string(img)
            text += extracted_text

    # Write all extracted text to the output file
    with open(txt_file, "w") as fp:
        fp.write(text)

    # Return the path to the created text file
    return txt_file


def text2json(
        txt_file: str,
        attribute_file: str = DEFAULT_ATTRIBUTE_LIST,
        instruction_file: str = DEFAULT_INSTRUCTIONS,
        model: str = DEFAULT_LLM_MODEL
) -> str:
    """
    Processes a text file and extracts attributes specified in an attribute file,
    following instructions in an instruction file, using a specified language model.

    Args:
        txt_file (str): The path to the text file from which to extract information.
        attribute_file (str, optional): The path to a file containing attributes to extract.
                                       Uses a default list if not specified.
        instruction_file (str, optional): The path to a file containing instructions for the extraction.
                                          Uses default instructions if not specified.
        model (str, optional): The language model to use for processing the text. Defaults to a pre-set model.

    Returns:
        str: The extracted attributes in JSON format, as generated by the model.

    Raises:
        FileNotFoundError: If the text file does not exist.
        OSError: If there are issues reading the attribute or instruction files.
    """

    # Read the entire text from the specified text file.
    with open(txt_file, "r") as fp:
        text_doc = fp.read()

    # Read the attributes if the attribute file exists, else default to an empty string.
    if attribute_file and os.path.exists(attribute_file):
        with open(attribute_file, "r") as fp:
            attributes = fp.read()
    else:
        attributes = ""

    # Read additional instructions if the instruction file exists, else default to an empty string.
    if instruction_file and os.path.exists(instruction_file):
        with open(instruction_file, "r") as fp:
            instructions = fp.read()
    else:
        instructions = ""

    # Access the API key from environment variables.
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Prepare the system prompt combining instructions and attributes.
    system_prompt = f"""
    Extract the following attributes from the text provided by the user.  In JSON format.
    {instructions}
    
    ## Attributes
    {attributes}
    """

    # Use the OpenAI API to create a chat completion using the specified model.
    # This sends the system prompt and the document text to the model.
    lease = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text_doc
            }
        ]
    )

    # Return the extracted information from the model's response.
    return lease.choices[0].message.content


def parse_arguments() -> argparse.Namespace:
    """
    Parses the command-line arguments provided to the script.

    Returns:
        argparse.Namespace: An object containing the values of the parsed arguments.

    This function sets up command-line options for a script that processes PDF files and extracts data using an LLM model.
    The user can specify various settings, such as the PDF URL, output settings, and configurations for the LLM model.

    Raises:
        argparse.ArgumentError: If there is an error parsing the arguments.
    """

    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Process and extract data from PDF files.")

    # Add mandatory URL argument
    parser.add_argument("url", type=str, help="URL of the PDF file where the PDF is located.")

    # Optional argument for the output file location
    parser.add_argument("--json", type=str, default=None,
                        help="Output file for the extracted attributes (default: standard output).")

    # Optional argument to specify a custom attribute list file
    parser.add_argument("--attributes", type=str, default=DEFAULT_ATTRIBUTE_LIST,
                        help=f"Path to the file containing a list of the attributes to extract "
                             f"(default: ./{DEFAULT_ATTRIBUTE_LIST}).")

    # Optional argument to specify which LLM model to use
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                        help=f"Which LLM model to use for processing (default: {DEFAULT_LLM_MODEL}).")

    # Optional argument for specifying the access key for the LLM model
    parser.add_argument("--model_key", type=str, default=None,
                        help="Access key for the LLM model if required.")

    # Optional argument for additional instructions to guide the LLM
    parser.add_argument("--instructions", type=str, default=DEFAULT_INSTRUCTIONS,
                        help=f"Additional instructions for the LLM (default: ./{DEFAULT_INSTRUCTIONS} if the file exists).")

    # Optional argument to specify the working directory
    parser.add_argument("--workdir", type=str, default=DEFAULT_WORKING_DIR,
                        help=f"Working directory for temporary files and outputs (default: ./{DEFAULT_WORKING_DIR}).")

    # Optional argument for specifying the location of the Tesseract OCR command
    parser.add_argument("--tesseract", type=str, default=None,
                        help=f"Location of the Tesseract command for OCR (default: {DEFAULT_TESSERACT_COMMAND}).")

    # Parse the arguments and return the result as a Namespace object
    try:
        return parser.parse_args()
    except argparse.ArgumentError as e:
        # If an error occurs during argument parsing, print the error message and exit
        print(e)
        sys.exit(1)


def requirements_satisfied(args) -> None:
    """
    Checks if all necessary requirements and dependencies are satisfied for the script to run properly.

    This function checks for the availability of an OpenAI API key and the Tesseract OCR command,
    both of which are critical for the script's operation.

    Args:
        args (argparse.Namespace): The parsed command-line arguments that might contain user-specified values.

    Raises:
        AttributeError: If any required configuration is missing or incorrect. Specifically, it checks for:
                        - OpenAI API key
                        - Tesseract OCR executable
    """

    # Check if an OpenAI API key is provided via command line or environment variable
    key = args.model_key or os.environ.get("OPENAI_API_KEY", None)
    if not key:
        raise AttributeError(
            "OpenAI key not specified (neither in environment variable OPENAI_API_KEY nor in --model_key argument)"
        )

    # Determine the path for the Tesseract command
    # Check if the default command path exists; otherwise, use the user-specified path or None
    tesseract = DEFAULT_TESSERACT_COMMAND if os.path.exists(DEFAULT_TESSERACT_COMMAND) else (
        args.tesseract if args.tesseract else None
    )

    # Raise an error if Tesseract command is not found at the specified or default path
    if not tesseract or not os.path.exists(tesseract):
        raise AttributeError(
            "Tesseract command not found.  (Install Tesseract and specify its path with --tesseract argument.)"
        )

    # Configure pytesseract to use the specified Tesseract command
    pytesseract.tesseract_cmd = tesseract


def main():
    """
    Main execution function for a script that downloads a PDF from a specified URL,
    extracts text from it, processes the text to extract attributes using an LLM,
    and optionally outputs the result to a file or standard output.
    """

    # Parse command-line arguments
    args = parse_arguments()

    # Ensure all system requirements and dependencies are satisfied
    try:
        requirements_satisfied(args)
    except AttributeError as e:
        print(f"Missing requirements: {e}")
        return

    # Ensure all system requirements and dependencies are satisfied
    if not os.path.exists(args.workdir):
        print(f"Working directory {args.workdir} does not exist.  Creating one.")
        os.makedirs(args.workdir)

    # Download the PDF from the specified URL to the working directory
    try:
        downloaded_pdf = download_pdf(args.url, save_to=args.workdir)
    except Exception as e:
        print(f"Failed to download PDF from {args.url}. {e}")
        return

    # Extract text from the downloaded PDF using OCR if necessary
    extracted_text = pdf2text(downloaded_pdf, working_dir=args.workdir)
    if not extracted_text:
        print(f"Failed to extract text from PDF {downloaded_pdf}")
        return

    # Convert the extracted text to JSON format using specified attributes and instructions
    try:
        json = text2json(
            extracted_text,
            attribute_file=args.attributes,
            instruction_file=args.instructions,
            model=args.model,
        )
    except Exception as e:
        print(f"Failed to extract information from {extracted_text}. {e}")
        return

    # If a JSON output file is specified, write the JSON to that file; otherwise, print to stdout
    try:
        if args.json:
            with open(args.json, "w") as fp:
                fp.write(json)
        else:
            print(json)
    except IOError as e:
        print(f"Failed to write to output file {args.json}. {e}")


if __name__ == '__main__':
    main()
