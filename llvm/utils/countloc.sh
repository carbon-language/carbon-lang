#!/bin/sh
##===- utils/countloc.sh - Counts Lines Of Code --------------*- Script -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This script finds all the source code files in the source code directories
# (excluding certain things), runs "wc -l" on them to get the number of lines in
# each file and then sums up and prints the total with awk. 
#
# The script takes no arguments but does expect to be run from somewhere in
# the top llvm source directory.
#
# Note that the implementation is based on llvmdo. See that script for more
# details.
##===----------------------------------------------------------------------===##

TOPDIR=`llvm-config --src-root`
if test -d "$TOPDIR" ; then
  cd $TOPDIR
  ./utils/llvmdo -dirs "include lib tools test utils examples" -code-only wc -l | awk '\
      BEGIN { loc=0; } \
      { loc += $1; } \
      END { print loc; }'
else
  echo "Can't find LLVM top directory in $TOPDIR"
fi
