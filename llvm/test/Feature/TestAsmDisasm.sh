#!/bin/sh
# test that every step outputs something that is consumable by 
# another step

rm -f test.bc.temp[12]

AS=$2/as
DIS=$2/dis

echo "======== Running assembler/disassembler test on $1"

# Two full cycles are needed for bitwise stability
(
  $AS  < $1      > $1.bc.1 || exit 1
  $DIS < $1.bc.1 > $1.ll.1 || exit 2
  $AS  < $1.ll.1 > $1.bc.2 || exit 3
  $DIS < $1.bc.2 > $1.ll.2 || exit 4

  diff $1.ll.[12] || exit 7

  # FIXME: When we sort things correctly and deterministically, we can
  # reenable this
  #diff $1.bc.[12] || exit 8

  rm $1.[bl][cl].[12]
  touch Output/$1.asmdis
) || ../Failure.sh "$1 ASM/DIS"

