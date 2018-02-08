# f18

## Selection of the C/C++ compiler

    export CXX=.../the/path/to/g++ 
    export CC=.../the/path/to/gcc    
   
## Installation of LLVM 5.0    
    
    ############ Extract LLVM, CLANG and other from git in current directory. 
    ############         
    ############ Question: 
    ############    Do we need the Clang sources for F18? 
    ############    Probably not but its nice to have the Clang source as 
    ############    example during development. 
    ############    Also, we need cland-format.
    ############
    ############        
    
    ROOT=$(pwd)
    REL=release_50
    
    # To build LLVM and Clang, we only need the head of the requested branch. 
    # Remove --single-branch --depth=1 if you want access to the whole repository

    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/llvm.git/       llvm

    # The following sub-repositories are probably not needed unless you also want to 
    # build clang: 

    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/clang.git/      llvm/tools/clang
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/openmp.git/     llvm/projects/openmp
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxx.git/     llvm/projects/libcxx
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxxabi.git/  llvm/projects/libcxxabi

    
    # List the version of all git sub-directories. They should all match $REL
    for dir in $(find "$ROOT/llvm" -name .git) ; do 
      cd $dir/.. ; 
      printf " %-15s %s\n" "$(git rev-parse --abbrev-ref HEAD)" "$(pwd)" ; 
    done
    
    
    ###########  Build LLVM & CLANG in $PREFIX 
    ###########  A Debug build can take a long time and a lot of disk space
    ###########  so I recommend making a Release build.
       
    PREFIX=$ROOT/usr
    mkdir $PREFIX
    
    mkdir $ROOT/llvm/build
    cd  $ROOT/llvm/build 
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
    make -j 4
    make install
    
## Installation of Flang 
 
    ######### Add $PREFIX/bin to PATH so that cmake finds llvm-config   
    
    export "PATH=$PREFIX/bin:$PATH"
    
    ######## Get Flang sources 
    git clone https://github.com/ThePortlandGroup/f18.git
    
    ######## Create a build directory for f18 
    mkdir $ROOT/f18-build
    cd $ROOT/f18-build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ../f18 
    make -j 4

    

