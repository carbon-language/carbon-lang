# f18

## Installation of LLVM 5.0    
    
    ############ Extract LLVM, CLANG and other from git in current directory. 
    ############         
    ############ Remark: 
    ############    Do we need the Clang sources for F18? 
    ############    Probably not but its nice to have the Clang source as 
    ############    example during development.
    ############
    ############        
    
    ROOT=$(pwd)
    REL=release_50
    
    git clone https://git.llvm.org/git/llvm.git/
    cd llvm/
    git checkout $REL
    
    cd $ROOT/llvm/tools
    git clone https://git.llvm.org/git/clang.git/
    git checkout $REL
    
    cd $ROOT/llvm/projects
    git clone https://git.llvm.org/git/openmp.git/ 
    cd openmp
    git checkout $REL
    
    cd $ROOT/llvm/projects
    git clone https://git.llvm.org/git/libcxx.git/
    cd libcxx
    git checkout $REL
    
    cd $ROOT/llvm/projects
    git clone https://git.llvm.org/git/libcxxabi.git/
    cd libcxxabi
    git checkout $REL
    
    # List the version of all git sub-directories 
    # They should all match $REL
    for dir in $(find "$ROOT" -name .git) ; do 
      cd $dir/.. ; 
      printf " %-15s %s\n" "$(git rev-parse --abbrev-ref HEAD)" "$(pwd)" ; 
    done
    
    
    ###########  Build LLVM & CLANG in $PREFIX 
        
    PREFIX=$ROOT/usr
    mkdir $PREFIX
    
    mkdir $ROOT/llvm/build
    cd  $ROOT/llvm/build 
    cmake CMAKE_INSTALL_PREFIX=$PREFIX 
    make -j 4
    make install
    
## 
    ######### Add $PREFIX/bin to PATH 
    
    
    
