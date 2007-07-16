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
if test ! -x "config.status" ; then
  # We must configure so build a list of configure options
  config_options="--prefix=$PREFIX --with-llvmgccdir=$PREFIX"
  config_options="$config_options $config_opts"
  echo ./configure $config_options
  ./configure $config_options || (echo "Can't configure llvm" ; exit 1)
fi

echo make $build_opts '&&' make install $build_opts
make $build_opts && make install $build_opts
