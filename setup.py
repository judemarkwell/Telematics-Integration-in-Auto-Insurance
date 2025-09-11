"""
Setup script for the Telematics Insurance System.

This script helps users set up the development environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "fastapi",
        "uvicorn[standard]",
        "numpy",
        "pandas", 
        "scikit-learn",
        "joblib",
        "pydantic",
        "python-dotenv",
        "httpx",
        "requests",
        "pytest",
        "pytest-asyncio"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True


def create_env_file():
    """Create .env file from example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from example...")
        try:
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created successfully")
            print("   Please review and update the configuration as needed")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No env.example file found, skipping .env creation")
    
    return True


def test_installation():
    """Test if the installation works."""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import fastapi
        import uvicorn
        import numpy
        import pandas
        import sklearn
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Test system
    if not run_command("python test_system.py", "Running system test"):
        return False
    
    return True


def main():
    """Main setup function."""
    print("🚗 Telematics Insurance System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Create environment file
    if not create_env_file():
        print("❌ Environment setup failed")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("🚀 You can now:")
    print("   • Run simulation: python main.py simulation")
    print("   • Start API server: python main.py api")
    print("   • View API docs: http://localhost:8000/docs")
    print("   • Run tests: python test_system.py")
    print("\n📚 See README.md for detailed usage instructions")


if __name__ == "__main__":
    main()

