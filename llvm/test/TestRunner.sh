#!/bin/sh
#
#  TestRunner.sh - This script is used to run arbitrary unit tests.  Unit
#  tests must contain the command used to run them in the input file, starting
#  immediately after a "RUN:" string.
#
#  This runner recognizes and replaces the following strings in the command:
#     %s   -  Replaced with the input name of the program
#

grep 'RUN:' $1 | sed "s/^.*RUN:\(.*\)$/\1/g;s/%s/$1/g" > Output/$1.script
OUTPUT=Output/$1.out


/bin/sh Output/$1.script > $OUTPUT 2>&1 || (
  echo "******************** TEST '$1' FAILED! ********************"
  echo "Command: "
  cat Output/$1.script
  echo "Output:"
  cat $OUTPUT
  rm $OUTPUT
  echo "******************** TEST '$1' FAILED! ********************"
)

