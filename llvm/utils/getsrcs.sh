#!/bin/sh
##===- utils/getsrcs.sh - Counts Lines Of Code ---------------*- Script -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Chris Lattner and Reid Spencer and is distributed 
# under the # University of Illinois Open Source License. See LICENSE.TXT for 
# details.
# 
##===----------------------------------------------------------------------===##
#
# This script just prints out the path names for all the source files in LLVM.
#
# Note that the implementation is based on llvmdo. See that script for more
# details.
##===----------------------------------------------------------------------===##

TOPDIR=`pwd | sed -e 's#\(.*/llvm\).*#\1#'`
if test -d "$TOPDIR" ; then
  cd $TOPDIR
  ./utils/llvmdo -dirs "include lib tools utils examples projects" echo
else
  echo "Can't find LLVM top directory in $TOPDIR"
fi
