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
#

FILENAME=$1
SUBST=$1
OUTPUT=$FILENAME.out

if test $# != 1; then
  # If more than one parameter is passed in, there must be three parameters:
  # The filename to read from (already processed), the command used to execute,
  # and the file to output to.
  SUBST=$2
  OUTPUT=$3
fi

SCRIPT=Output/$OUTPUT.script
grep 'RUN:' $FILENAME | sed "s|^.*RUN:\(.*\)$|\1|g;s|%s|$SUBST|g" > $SCRIPT


/bin/sh $SCRIPT > $OUTPUT 2>&1 || (
  echo "******************** TEST '$FILENAME' FAILED! ********************"
  echo "Command: "
  cat $SCRIPT
  echo "Output:"
  cat $OUTPUT
  rm $OUTPUT
  echo "******************** TEST '$FILENAME' FAILED! ********************"
)

