#!/bin/sh
#
#  TestRunner.sh - This script is used to run arbitrary unit tests.  Unit
#  tests must contain the command used to run them in the input file, starting
#  immediately after a "RUN:" string.
#
#  This runner recognizes and replaces the following strings in the command:
#
#     %s - Replaced with the input name of the program, or the program to
#          execute, as appropriate.
#     %llvmgcc - llvm-gcc command
#     %llvmgxx - llvm-g++ command
#     %prcontext - prcontext.tcl script
#
TESTFILE=$1
if test `dirname $TESTFILE` == . ; then
  TESTPATH=`pwd`
  SUBDIR=""
  while test `basename $TESTPATH` != "test" -a ! -z "$TESTPATH" ; do
    tmp=`basename $TESTPATH`
    SUBDIR="$tmp/$SUBDIR"
    TESTPATH=`dirname $TESTPATH`
  done
  if test -d "$TESTPATH" ; then
    cd $TESTPATH
    make check-one TESTONE="$SUBDIR$TESTFILE"
  else
    echo "Can't find llvm/test directory in " `pwd`
  fi
else
  make check-one TESTONE=$TESTFILE
fi
