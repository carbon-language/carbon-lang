#!/bin/sh

#
# Split the current LD_LIBRARY_PATH into two separate components.
#
FirstLDP=`echo $LD_LIBRARY_PATH | cut -d\; -f1`
SecondLDP=`echo $LD_LIBRARY_PATH | cut -d\; -f2`

#
# Now create a new LD_LIBRARY_PATH with our command line options as
# the first section.
#
LD_LIBRARY_PATH="$3:${FirstLDP}\;${SecondLDP}"
export LD_LIBRARY_PATH

AS=$2/as
DIS=$2/dis
OPT=$2/opt

echo "======== Running optimizer test on $1"

(
  $AS < $1 | $OPT -q -inline -dce -constprop -dce |$DIS| $AS > $1.bc.1 || exit 1

  # Should not be able to optimize further!
  $OPT -q -constprop -dce < $1.bc.1 > $1.bc.2 || exit 2

  $DIS < $1.bc.1 > $1.ll.1 || exit 3
  $DIS < $1.bc.2 > $1.ll.2 || exit 3
  gdiff -u $1.ll.[12] || exit 3

  # Try out SCCP 
  $AS < $1 | $OPT -q -inline -dce -sccp -dce \
           | $DIS | $AS > $1.bc.3 || exit 1

  # Should not be able to optimize further!
  $OPT -q -sccp -dce < $1.bc.3 > $1.bc.4 || exit 2
  $DIS < $1.bc.3 > $1.ll.3 || exit 3
  $DIS < $1.bc.4 > $1.ll.4 || exit 3
  gdiff -u $1.ll.[34] || exit 3
  rm $1.bc.[1234] $1.ll.[1234]
  
  touch Output/$1.opt  # Success!
)|| ../Failure.sh "$1 Optimizer"
