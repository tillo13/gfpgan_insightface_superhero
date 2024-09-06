# Insightface/GFPGAN Superhero Demo

Welcome to the Insightface Superhero Demo. This project aims to demonstrate the process of creating a superhero image by swapping faces using Insightface, followed by enhancing the image using GFPGAN for better quality.

## Project Structure

The project is organized in the following directory structure:

```
.  
│
├── create_superhero.py
├── enhance_image_via_import.py
├── gather_pythons.py
├─── __pycache__
├─── comparisons
├─── gfpgan
│    └── weights
├─── output
├─── utilities_needed
│    └── gfpgan
│        └── GFPGAN-master
│             ├── .github
│             │    └── workflows
│             ├── .vscode
│             ├── assets
│             ├── experiments
│             │    └── pretrained_models
│             ├── gfpgan
│             │    ├── archs
│             │    ├── data
│             │    ├── models
│             │    └── weights
│             ├── inputs
│             │    ├── cropped_faces
│             │    └── whole_imgs
│             ├── options
│             ├── scripts
│             └── tests
│                  └── data
│                       ├── ffhq_gt.lmdb
│                       └── gt
└─── data
```

## Files

There are a total of 29 Python files spread across different directories. Key files include:

- `create_superhero.py`: Main script to create a superhero image by swapping faces and enhancing the image.
- `enhance_image_via_import.py`: Script for custom image enhancement after using GFPGAN.
- `gather_pythons.py`: Script to gather all `.py` files in the project directory and document them.
- Various scripts and modules in `utilities_needed/gfpgan/GFPGAN-master` for GFPGAN operations and testing.

## Installation

### Prerequisites

- Python 3.6 or later
- pip

### Setup

1. Clone the repository.
    ```bash
    git clone https://github.com/your-repository/insightface-superhero-demo.git
    cd insightface-superhero-demo
    ```

2. Create a virtual environment.
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment.

    On Windows:
    ```bash
    venv\Scripts\activate
    ```

    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

4. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Create Superhero Image

1. Place the input image as `person.png` and the superhero pose image as `superhero.png` in the project root directory.
2. Run the `create_superhero.py` script.
    ```bash
    python create_superhero.py
    ```

### Enhance Image

The initial enhancement is performed by GFPGAN via the `create_superhero.py` script. Further custom enhancements can be applied using the `enhance_image_via_import.py` script.

### Collect all Python files

Run `gather_pythons.py` to collect and document all Python files in the directory.

```bash
python gather_pythons.py
```

## About the Main Script (`create_superhero.py`)

This script performs the following operations:

1. Sets up the environment by creating necessary directories and downloading required models.
2. Initializes the InsightFace face analysis and face swapping models.
3. Loads and detects faces from the input images.
4. Swaps faces between the source and target images.
5. Enhances the resulting image using GFPGAN.
6. Optionally applies custom enhancements on the GFPGAN-enhanced image.

### Example

```bash
cd insightface-superhero-demo
python create_superhero.py
```

### Workflow of `create_superhero.py`

1. Setup environment
2. Initialize InsightFace
3. Initialize GFPGAN
4. Detect faces in the input images
5. Swap faces
6. Enhance the swapped image
7. Apply custom enhancements
8. Compare images and log the results

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.