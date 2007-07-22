#!/bin/sh

. ../library.sh

process_arguments "$@"

# See if we have previously been configured by sensing the presense
# of the config.status scripts
config_status="$build_dir/config.status"
if test ! -d "$config_status" ; then
  # We must configure so build a list of configure options
  config_options="--prefix=$PREFIX --with-llvm-top=$LLVM_TOP"
  echo ./configure $config_options
  ./configure $config_options
fi

make
