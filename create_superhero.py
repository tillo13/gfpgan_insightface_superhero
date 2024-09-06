import os
import urllib.request
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
from tqdm import tqdm
import logging
from enhance_image_via_import import enhance_image
import sys

# Constants
INCOMING_IMAGE_PATH = 'person.png'
POSE_IMAGE_PATH = 'superhero.png'
OUTPUT_DIR = 'output'
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'output_image.png')
OUTPUT_INITIAL_GFPGAN_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'output_image_initial_gfpgan.png')
OUTPUT_ENHANCED_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'output_image_initial_gfpgan_enhanced.png')
MODEL_URL = 'https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'
GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
LANDMARKS_MODEL_PATH = os.path.join('utilities_needed', 'shape_predictor_68_face_landmarks.dat')
LANDMARKS_MODEL_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'
MODEL_LOCAL_PATH = os.path.join('utilities_needed', 'inswapper_128.onnx')
GFPGAN_MODEL_PATH = os.path.join('utilities_needed', 'GFPGANv1.4.pth')
GFPGAN_CODE_PATH = os.path.join('utilities_needed', 'gfpgan')
FACE_ANALYSIS_MODEL_NAME = 'buffalo_l'
UTILITIES_DIRECTORY = 'utilities_needed'

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, path):
    """Download a file from a URL if it does not already exist"""
    if not os.path.exists(path):
        logger.info(f"Downloading the file from {url}...")
        with tqdm(total=100, desc=f"Downloading {os.path.basename(path)}", unit='%', ncols=80) as pbar:
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress = round((block_num * block_size) / total_size * 100, 2)
                    pbar.n = progress
                    pbar.refresh()
            urllib.request.urlretrieve(url, path, reporthook)
        logger.info(f"File downloaded successfully at {path}.")
    else:
        logger.info(f"File already exists at {path}. Skipping download.")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a specific directory"""
    from zipfile import ZipFile
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extracted {zip_path} to {extract_to}.")

def setup_environment():
    """Setup the environment by ensuring all necessary directories and downloading required models"""
    if not os.path.exists(UTILITIES_DIRECTORY):
        os.makedirs(UTILITIES_DIRECTORY)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Download required models and files
    logger.info("Setting up environment...")
    download_file(GFPGAN_MODEL_URL, GFPGAN_MODEL_PATH)
    download_file(MODEL_URL, MODEL_LOCAL_PATH)
    # Download and extract GFPGAN code only if it does not exist
    gfpgan_code_zip_path = os.path.join(UTILITIES_DIRECTORY, 'GFPGAN.zip')
    if not os.path.exists(GFPGAN_CODE_PATH):
        download_file('https://github.com/TencentARC/GFPGAN/archive/refs/heads/master.zip', gfpgan_code_zip_path)
        extract_zip(gfpgan_code_zip_path, GFPGAN_CODE_PATH)

    # Download landmarks model only if it does not exist
    if not os.path.exists(LANDMARKS_MODEL_PATH):
        download_file(LANDMARKS_MODEL_URL, LANDMARKS_MODEL_PATH)
    
    logger.info("Environment setup completed.")

# Call setup_environment to ensure all files are downloaded
setup_environment()

# Ensure the GFPGAN code path is in sys.path
if GFPGAN_CODE_PATH not in sys.path:
    sys.path.append(GFPGAN_CODE_PATH)
from gfpgan import GFPGANer

# Initialize InsightFace
logger.info("Initializing InsightFace...")
face_analysis = insightface.app.FaceAnalysis(name=FACE_ANALYSIS_MODEL_NAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analysis.prepare(ctx_id=0, det_size=(640, 640))
logger.info("InsightFace initialized.")

# Initialize the face swapping model
logger.info("Initializing face swapper...")
face_swapper = insightface.model_zoo.get_model(MODEL_LOCAL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
logger.info("Face swapper initialized.")

# Initialize the GFPGAN face enhancer
logger.info("Initializing GFPGAN...")
gfpganer = GFPGANer(model_path=GFPGAN_MODEL_PATH, upscale=2, arch='clean', channel_multiplier=2)
logger.info("GFPGAN initialized.")

def detect_face(image_path):
    """Detect a face in the image at image_path using InsightFace"""
    logger.info(f"Detecting face in image: {image_path}...")
    image = cv2.imread(image_path)
    faces = face_analysis.get(image)
    if faces:
        logger.info(f"Face detected in image: {image_path}")
        return faces[0], image
    else:
        raise ValueError(f"No face detected in image: {image_path}")

def swap_faces(source_face, target_face, target_image):
    """Swap faces between source and target images"""
    logger.info("Swapping faces in the target image...")
    return face_swapper.get(target_image, target_face, source_face, paste_back=True)

def enhance_image_gfpgan(image):
    """Enhance the image using GFPGAN"""
    logger.info("Enhancing image using GFPGAN...")
    _, _, enhanced_image = gfpganer.enhance(image, has_aligned=False, only_center_face=True, paste_back=True)
    return enhanced_image

def images_difference(image_path1, image_path2):
    """Compare images and return the number of differing pixels"""
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    # Ensure the images are the same size before comparison
    if img1.shape == img2.shape:
        difference = cv2.absdiff(img1, img2)
        non_zero_count = np.count_nonzero(difference)
        return non_zero_count
    else:
        return -1  # Different sizes, cannot compare

def main():
    try:
        # Load and detect face from incoming image
        source_face, source_image = detect_face(INCOMING_IMAGE_PATH)
        logger.info("Detected source face.")

        # Load and detect face from pose image
        target_face, target_image = detect_face(POSE_IMAGE_PATH)
        logger.info("Detected target face.")

        # Perform the face swap
        logger.info("Performing face swap...")
        output_image = swap_faces(source_face, target_face, target_image)
        cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
        logger.info(f"Face swap completed and saved as {OUTPUT_IMAGE_PATH}")

        # Enhance the output image through GFPGAN
        logger.info("Enhancing output image using GFPGAN...")
        enhanced_image = enhance_image_gfpgan(output_image)
        cv2.imwrite(OUTPUT_INITIAL_GFPGAN_IMAGE_PATH, enhanced_image)
        logger.info(f"Enhanced output image saved as {OUTPUT_INITIAL_GFPGAN_IMAGE_PATH}")

        # Apply our custom enhancement on the GFPGAN enhanced image
        logger.info("Applying custom enhancements to the initial GFPGAN enhanced image...")
        final_image = enhance_image(OUTPUT_INITIAL_GFPGAN_IMAGE_PATH, OUTPUT_DIR)
        
        if final_image is None:
            logger.error("Error enhancing the image using custom enhancements.")
        else:
            logger.info("Custom enhancements applied successfully.")
            final_image.save(OUTPUT_ENHANCED_IMAGE_PATH)
            logger.info(f"Final enhanced image saved as {OUTPUT_ENHANCED_IMAGE_PATH}")

            # Compare images
            diff = images_difference(OUTPUT_INITIAL_GFPGAN_IMAGE_PATH, OUTPUT_ENHANCED_IMAGE_PATH)
            if diff == -1:
                logger.info("Initial GFPGAN enhanced and final images are of different sizes and cannot be directly compared.")
            elif diff == 0:
                logger.info("Initial GFPGAN enhanced and final images are identical.")
            else:
                logger.info(f"Initial GFPGAN enhanced and final images have {diff} differing pixels.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()