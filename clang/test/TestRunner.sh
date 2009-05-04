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
#     %S - Replaced with the directory where the input file resides
#     %prcontext - prcontext.tcl script
#     %t - temporary file name (derived from testcase name)
#

FILENAME=$1
TESTNAME=$1
SUBST=$1
FILEDIR=`dirname $TESTNAME`

# Make diagnostic printing more determinstic.
export COLUMNS=0

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

# Resolve the path, and Make sure $CLANG actually exists; otherwise
# ensuing failures are non-obvious.
CLANG=$(which "$CLANG")
if [ -z $CLANG ]; then
  echo "Couldn't find 'clang' program, try setting CLANG in your environment"
  exit 1
fi

if [ -n "$VG" ]; then
  rm -f $OUTPUT.vg
  CLANG="valgrind --leak-check=full --quiet --log-file=$OUTPUT.vg $CLANG"
fi

# Assuming $CLANG is correct, use it to derive clang-cc. We expect to
# be looking in a build directory, so just add '-cc'.
CLANGCC=$CLANGCC
if [ ! -n "$CLANGCC" ]; then
    CLANGCC="$CLANG-cc"
fi

# Try to sanity check $CLANGCC too
CLANGCC=$(which "$CLANGCC")
# If that failed, ask clang.
if [ -z "$CLANGCC" ]; then
    CLANGCC=$($CLANG -print-prog-name=clang-cc)
fi
if [ -z "$CLANGCC" ]; then
  echo "Couldn't find 'clang-cc' program, make sure clang is found in your build directory"
  exit 1
fi

SCRIPT=$OUTPUT.script
TEMPOUTPUT=$OUTPUT.tmp
grep 'RUN:' $FILENAME | \
  sed -e "s|^.*RUN:\(.*\)$|\1|g" \
      -e "s| clang | $CLANG |g" \
      -e "s| clang-cc | $CLANGCC |g" \
      -e "s|%s|$SUBST|g" \
      -e "s|%S|$FILEDIR|g" \
      -e "s|%prcontext|prcontext.tcl|g" \
      -e "s|%t|$TEMPOUTPUT|g" > $SCRIPT

IS_XFAIL=0
if (grep -q XFAIL $FILENAME); then
    IS_XFAIL=1
    printf "XFAILED '$TESTNAME': "
    grep XFAIL $FILENAME
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
