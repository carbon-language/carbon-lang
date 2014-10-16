#!/bin/sh -xe

llvm_components="\
all-targets \
analysis \
asmparser \
asmprinter \
bitreader \
bitwriter \
codegen \
core \
debuginfo \
executionengine \
instrumentation \
interpreter \
ipo \
irreader \
linker \
mc \
mcjit \
objcarcopts \
option \
profiledata \
scalaropts \
support \
target \
"

if [ "$1" == "--print-components" ] ; then
  echo $llvm_components
  exit 0
fi

gollvmdir=$(dirname "$0")/llvm

workdir=$gollvmdir/workdir
llvmdir=$gollvmdir/../../..
llvm_builddir=$workdir/llvm_build

mkdir -p $llvm_builddir

cmake_flags="../../../../.. $@"
llvm_config="$llvm_builddir/bin/llvm-config"

if test -n "`which ninja`" ; then
  # If Ninja is available, we can speed up the build by building only the
  # required subset of LLVM.
  (cd $llvm_builddir && cmake -G Ninja $cmake_flags)
  ninja -C $llvm_builddir llvm-config
  llvm_buildtargets="$($llvm_config --libs $llvm_components | sed -e 's/-l//g')"
  ninja -C $llvm_builddir $llvm_buildtargets FileCheck
else
  (cd $llvm_builddir && cmake $cmake_flags)
  make -C $llvm_builddir -j4
fi

llvm_version="$($llvm_config --version)"
llvm_cflags="$($llvm_config --cppflags)"
llvm_ldflags="$($llvm_config --ldflags) $($llvm_config --libs $llvm_components) $($llvm_config --system-libs)"
if [ $(uname) != "Darwin" ]; then
  # OS X doesn't like -rpath with cgo. See:
  # https://code.google.com/p/go/issues/detail?id=7293
  llvm_ldflags="-Wl,-rpath,$($llvm_config --libdir) $llvm_ldflags"
fi
sed -e "s#@LLVM_CFLAGS@#$llvm_cflags#g; s#@LLVM_LDFLAGS@#$llvm_ldflags#g" $gollvmdir/llvm_config.go.in > \
  $gollvmdir/llvm_config.go
printf "package llvm\n\nconst Version = \"%s\"\n" "$llvm_version" > $gollvmdir/version.go
