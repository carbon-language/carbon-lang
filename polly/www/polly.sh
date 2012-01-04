#!/bin/bash -xe

export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
export POLLY_SRC=${LLVM_SRC}/tools/polly
export CLOOG_SRC=${BASE}/cloog_src
export CLOOG_INSTALL=${BASE}/cloog_install
export LLVM_BUILD=${BASE}/llvm_build
export SCOPLIB_DIR=${BASE}/scoplib-0.2.0
export POCC_DIR=${BASE}/pocc-1.0-rc3.1

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

${POLLY_SRC}/utils/checkout_cloog.sh ${CLOOG_SRC}
cd ${CLOOG_SRC}

if ! test -e ${CLOOG_SRC}/config.log; then
    ./configure --prefix=${CLOOG_INSTALL}
fi
make
make install
cd ${BASE}

if ! test -d ${POCC_DIR}; then
    wget http://www.cse.ohio-state.edu/~pouchet/software/pocc/download/pocc-1.0-rc3.1-full.tar.gz
    tar xzf pocc-1.0-rc3.1-full.tar.gz
    cd ${POCC_DIR}
    ./install.sh
    cd ${BASE}
fi
export PATH=${POCC_DIR}/bin:$PATH

if ! test -d ${SCOPLIB_DIR}; then
    wget http://www.cse.ohio-state.edu/~pouchet/software/pocc/download/modules/scoplib-0.2.0.tar.gz
    tar xzf  scoplib-0.2.0.tar.gz
    cd ${SCOPLIB_DIR}
    ./configure --enable-mp-version --prefix=${SCOPLIB_DIR}/usr
    make -j${procs} -l${procs} && make install
fi

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

if which cmake ; then
    cmake -DCMAKE_PREFIX_PATH=${CLOOG_INSTALL} ${LLVM_SRC}
    make -j$procs -l$procs
    make polly-test
else
    ${LLVM_SRC}/configure --with-cloog=${CLOOG_INSTALL} --with-isl=${CLOOG_INSTALL}
    make -j$procs -l$procs
    make polly-test -C tools/polly/test/
fi
