#!/bin/bash -xe

export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
export POLLY_SRC=${LLVM_SRC}/tools/polly
export CLANG_SRC=${LLVM_SRC}/tools/clang
export ISL_SRC=${BASE}/isl_src
export ISL_INSTALL=${BASE}/isl_install
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

${POLLY_SRC}/utils/checkout_isl.sh ${ISL_SRC}
cd ${ISL_SRC}

if ! test -e ${ISL_SRC}/config.log; then
    ./configure --prefix=${ISL_INSTALL}
fi
make
make install
cd ${BASE}

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

if which cmake ; then
    cmake -DCMAKE_PREFIX_PATH=${ISL_INSTALL} ${LLVM_SRC}
    make -j$procs -l$procs
    make check-polly
else
    ${LLVM_SRC}/configure --with-isl=${ISL_INSTALL}
    make -j$procs -l$procs
    make check-polly -C tools/polly/test/
fi
