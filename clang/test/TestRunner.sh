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
#     %t - temporary file name (derived from testcase name)
#

FILENAME=$1
TESTNAME=$1
SUBST=$1

OUTPUT=Output/$1.out

# create the output directory if it does not already exist
mkdir -p `dirname $OUTPUT` > /dev/null 2>&1

if test $# != 1; then
  # If more than one parameter is passed in, there must be three parameters:
  # The filename to read from (already processed), the command used to execute,
  # and the file to output to.
  SUBST=$2
  OUTPUT=$3
  TESTNAME=$3
fi

ulimit -t 40

# Verify the script contains a run line.
grep -q 'RUN:' $FILENAME || ( 
   echo "******************** TEST '$TESTNAME' HAS NO RUN LINE! ********************"
   exit 1
)

# Run under valgrind if the VG environment variable has been set.
CLANG=$CLANG
if [ ! -n "$CLANG" ]; then
    CLANG="clang"
fi
if [ -n "$VG" ]; then
  rm -f $OUTPUT.vg
  CLANG="valgrind --leak-check=full --quiet --log-file=$OUTPUT.vg $CLANG"
fi

SCRIPT=$OUTPUT.script
TEMPOUTPUT=$OUTPUT.tmp
grep 'RUN:' $FILENAME | \
  sed -e "s|^.*RUN:\(.*\)$|\1|g" \
      -e "s|clang|$CLANG|g" \
      -e "s|%s|$SUBST|g" \
      -e "s|%llvmgcc|llvm-gcc -emit-llvm|g" \
      -e "s|%llvmgxx|llvm-g++ -emit-llvm|g" \
      -e "s|%prcontext|prcontext.tcl|g" \
      -e "s|%t|$TEMPOUTPUT|g" > $SCRIPT

IS_XFAIL=0
if (grep -q XFAIL $FILENAME); then
    IS_XFAIL=1
    printf "XFAILED '$TESTNAME': "
    grep XFAIL $FILENAME
fi

if (grep -q "%llvmgcc" $FILENAME); then
  if [ -z "$(llvm-gcc --version 2> /dev/null)" ]; then
    IS_XFAIL=1
    echo "llvm-gcc not found"
  fi
fi

if (grep -q "%llvmgxx" $FILENAME); then
  if [ -z "$(llvm-g++ --version 2> /dev/null)" ]; then
    IS_XFAIL=1
    echo "llvm-g++ not found"
  fi
fi

/bin/sh $SCRIPT > $OUTPUT 2>&1
SCRIPT_STATUS=$?

if [ -n "$VG" ]; then
  [ ! -s $OUTPUT.vg ]
  VG_STATUS=$?
else
  VG_STATUS=0
fi

if [ $IS_XFAIL -ne 0 ]; then
    if [ $SCRIPT_STATUS -ne 0 ]; then
        SCRIPT_STATUS=0
    else
        SCRIPT_STATUS=1
    fi
fi

if [ $SCRIPT_STATUS -ne 0 -o $VG_STATUS -ne 0 ]; then
  echo "******************** TEST '$TESTNAME' FAILED! ********************"
  echo "Command: "
  cat $SCRIPT
  if [ $SCRIPT_STATUS -eq 0 ]; then
    echo "Output:"
  elif [ $IS_XFAIL -ne 0 ]; then
    echo "Incorrect Output (Expected Failure):"
  else
    echo "Incorrect Output:"
  fi
  cat $OUTPUT
  if [ $VG_STATUS -ne 0 ]; then
    echo "Valgrind Output:"
    cat $OUTPUT.vg
  fi
  echo "******************** TEST '$TESTNAME' FAILED! ********************"
  exit 1
fi

