#!/bin/sh
#
#  TestRunner.sh - This script is used to run arbitrary unit tests.  Unit
#  tests must contain the command used to run them in the input file, starting
#  immediately after a "RUN:" string.
#
#  This runner recognizes and replaces the following strings in the command:
#     %s   -  Replaced with the input name of the program
#

PROGRAM=`grep 'RUN:' $1 | sed "s/^.*RUN:\(.*\)$/\1/g;s/%s/$1/g" | head -n 1`
OUTPUT=Output/$1.out

($PROGRAM) > $OUTPUT 2>&1 || (
  echo "******************** TEST '$1' FAILED! ********************"
  echo "Command: " $PROGRAM
  echo "Output:"
  cat $OUTPUT
  rm $OUTPUT
  echo "******************** TEST '$1' FAILED! ********************"
)

