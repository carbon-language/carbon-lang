#!/bin/bash -xe

export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
export POLLY_SRC=${LLVM_SRC}/tools/polly
export CLANG_SRC=${LLVM_SRC}/tools/clang
export LLVM_BUILD=${BASE}/llvm_build

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

if ! test -d ${LLVM_SRC}; then
    git clone http://llvm.org/git/llvm.git ${LLVM_SRC}
fi

if ! test -d ${POLLY_SRC}; then
    git clone http://llvm.org/git/polly.git ${POLLY_SRC}
fi

if ! test -d ${CLANG_SRC}; then
    git clone http://llvm.org/git/clang.git ${CLANG_SRC}
fi

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

if which cmake ; then
    cmake ${LLVM_SRC}
    make -j$procs -l$procs
    make check-polly
else
    ${LLVM_SRC}/configure
    make -j$procs -l$procs
    make check-polly -C tools/polly/test/
fi
