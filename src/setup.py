import sys
from cx_Freeze import setup, Executable

# Determine the base (GUI or console)
base = "Win32GUI" if sys.platform == "win32" else None

# Define build options
build_exe_options = {
    "packages": [
        "mediapipe",
        "cv2",
        "numpy",
        "matplotlib",
        "scipy",
        "PIL",
        "dataclasses"  # For your @dataclass usage
    ],
    "include_files": [
        # Explicitly include MediaPipe and OpenCV files
        # Replace these paths with the actual paths in your virtual environment
        (f"venv/Lib/site-packages/mediapipe", "mediapipe"),
        (f"venv/Lib/site-packages/cv2", "cv2")
    ],
    # Add any additional path hints for binaries
    "bin_path_includes": [
        f"venv/Lib/site-packages/mediapipe/library",
        f"venv/Lib/site-packages/cv2"
    ]
}

# Setup configuration
setup(
    name="HandDrawingApp",
    version="1.0",
    description="Hand Tracking Drawing Application",
    options={"build_exe": build_exe_options},
    executables=[Executable(
        "src\main.py",
        base=base,
        icon="D:\Python\WhiteboardAI\src\icon.ico"  # Optional: add an .ico file path if you want a custom icon
    )]
)