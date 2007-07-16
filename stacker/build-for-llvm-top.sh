#!/bin/sh

is_debug=1
for arg in "$@" ; do
  case "$arg" in
    LLVM_TOP=*)
      LLVM_TOP=`echo "$arg" | sed -e 's/LLVM_TOP=//'`
      ;;
    PREFIX=*)
      PREFIX=`echo "$arg" | sed -e 's/PREFIX=//'`
      ;;
    ENABLE_OPTIMIZED=1)
      is_debug=0
      ;;
    *=*)
      build_opts="$build_opts $arg"
      ;;
    --*)
      config_opts="$config_opts $arg"
      ;;
  esac
done

# See if we have previously been configured by sensing the presense
# of the config.status scripts
config_status="$build_dir/config.status"
if test ! -d "$config_status" ; then
  # We must configure so build a list of configure options
  config_options="--prefix=$PREFIX --with-llvmsrc=$LLVM_TOP/llvm"
  config_options="$config_options --with-llvmobj=$LLVM_TOP/llvm"
  echo ./configure $config_options $config_opts
  ./configure $config_options $config_opts
fi

make $build_opts && make install $build_opts
