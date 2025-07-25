#!/bin/bash
curdir=$PWD
mydir="${0%/*}"
version=$1
cd $mydir

# Detect system
# Get OS name, and arhitecture
os=$(uname -s)
arch=$(uname -m)

echo "Downloading PyTorch"

# Choose the correct URL based on system
if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    # macOS ARM64 
    # Apple Silicon Macs (M1/M2/M3)
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-$version.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
elif [[ "$os" == "Darwin" ]]; then
    # macOS Intel
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-$version.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
else
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
fi

echo "Unzipping PyTorch"
unzip libtorch-$version.zip >> /dev/null
mv libtorch libtorch-$version

echo "Successfully installed PyTorch $version for $os $arch"


# Check multiple locations for OpenMP (Open Multi-Processing)
#       An API that allows for multiple CPU cores to be used simultaneously
# PyTorch uses OpenMP for parallel computation
#       For matrix and tensor operations       
# Without OpenMP was getting "dyld: Library not loaded: @rpath/libomp.dylib" error, when running the test file
#       dyld is macOS's system component which loads shared libraries, resolves symbol dependencies between libraries and manages library paths + versions
# To resolve I copied the location of where OpenMP was to where Torch expected it to be (ie the commands at the bottom of file)  

# Can this be done more efficiently / or is this needed at all (ie is there a better way to either avoid the error all together or resolve the error)

if [[ "$os" == "Darwin" ]]; then
    echo "Checking OpenMP dependency for macOS..."
    
    # Check if OpenMP is already available
    openmp_found=false
    openmp_paths=(
        "/opt/homebrew/lib/libomp.dylib"     
        "/usr/local/lib/libomp.dylib"       
        "$CONDA_PREFIX/lib/libomp.dylib"  
    )
    
    for path in "${openmp_paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "OpenMP found at: $path"
            openmp_found=true
            break
        fi
    done
    
    # Attempt at making robust for systems (Not sure if this works, my device had installed via Conda)
    if [[ "$openmp_found" == false ]]; then
        echo "OpenMP not found. PyTorch requires OpenMP to run properly."
        
        if command -v brew &> /dev/null; then
            echo "Installing OpenMP via Homebrew..."
            if brew install libomp; then
                echo "OpenMP installed successfully via Homebrew"
            else
                echo "Failed to install OpenMP via Homebrew"
                echo "Install manually: brew install libomp"
            fi
        elif command -v conda &> /dev/null; then
            echo "Homebrew not found. Attempting to install via conda..."
            if conda install -c conda-forge llvm-openmp -y; then
                echo "OpenMP installed successfully via conda"
            else
                echo "Failed to install OpenMP via conda"
                echo "Install manually: conda install -c conda-forge llvm-openmp"
            fi
        else
            echo "Neither Homebrew nor conda found."
            echo "Install OpenMP manually:"
            echo "Option 1: Install Homebrew, then run: brew install libomp"
            echo "Option 2: Install conda, then run: conda install -c conda-forge llvm-openmp"
        fi
    fi
fi

# Command I used to resolve issue once path to libomp was found
if [[ "$os" == "Darwin" && -f "/opt/anaconda3/lib/libomp.dylib" ]]; then
    echo "Copying OpenMP library to PyTorch directory..."
    cp /opt/anaconda3/lib/libomp.dylib libtorch-$version/lib/
    echo "OpenMP library copied successfully"
fi

cd $curdir