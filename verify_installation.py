#!/usr/bin/env python3
"""
Verify that the installation is correctly set up for your platform.

This script checks:
- Python version
- PyTorch installation
- Available acceleration (CUDA, MPS, or CPU)
- All required dependencies
"""

import sys
from importlib.metadata import version


def check_python_version():
    """Check if Python version meets requirements."""
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)

    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version >= (3, 11):
        print("✓ Python version meets requirements (>=3.11)")
        return True
    else:
        print("✗ Python version too old. Please upgrade to Python 3.11 or higher.")
        return False


def check_pytorch():
    """Check PyTorch installation and available acceleration."""
    print("\n" + "=" * 60)
    print("PyTorch Installation Check")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  - Number of GPUs: {torch.cuda.device_count()}")
            device_type = "cuda"
        # Check MPS
        elif torch.backends.mps.is_available():
            print(f"✓ MPS is available (Apple Silicon GPU acceleration)")
            device_type = "mps"
        # CPU only
        else:
            print(f"ℹ No GPU acceleration available, using CPU")
            device_type = "cpu"

        # Test device
        print(f"\nTesting {device_type.upper()} device...")
        device = torch.device(device_type)
        test_tensor = torch.randn(10, 10).to(device)
        result = test_tensor @ test_tensor.T
        print(f"✓ Successfully performed computation on {device_type.upper()}")

        return True

    except ImportError:
        print("✗ PyTorch is not installed")
        print("  Install with: uv pip install torch torchvision")
        return False
    except Exception as e:
        print(f"✗ Error testing PyTorch: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Dependencies Check")
    print("=" * 60)

    required_packages = [
        "torch",
        "torchvision",
        "cv2",  # opencv-python
        "numpy",
        "scipy",
        "skimage",  # scikit-image
        "matplotlib",
        "pandas",
        "yaml",  # pyyaml
        "tensorboard",
        "tqdm",
        "kornia",
    ]

    package_names = {
        "cv2": "opencv-python",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
    }

    all_installed = True

    for package in required_packages:
        try:
            pkg_name = package_names.get(package, package)
            __import__(package)
            try:
                ver = version(pkg_name)
                print(f"✓ {pkg_name}: {ver}")
            except:
                print(f"✓ {pkg_name}: installed")
        except ImportError:
            print(f"✗ {package_names.get(package, package)}: not installed")
            all_installed = False

    return all_installed


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("STAGING MODEL INSTALLATION VERIFICATION")
    print("=" * 60)

    checks = [
        ("Python version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
    ]

    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"\n✗ Error checking {name}: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all(results):
        print("✓ All checks passed! Installation is complete and ready to use.")
        print("\nYou can now run:")
        print("  python main.py --mode train")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nFor installation help, see README.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
