#!/bin/sh
# This is useful because it prints out all of the source files.  Useful for
# greps.
TOPDIR=`pwd | sed -e 's#\(.*/llvm\).*#\1#'`
if test -d "$TOPDIR" ; then
  cd $TOPDIR
  ./utils/llvmdo -dirs "include lib tools utils examples projects" echo
else
  echo "Can't find LLVM top directory in $TOPDIR"
fi
