#!/bin/sh
# test that every step outputs something that is consumable by 
# another step

rm -f test.bc.temp[12]

LD_LIBRARY_PATH=../lib/Assembly/Parser/Debug:../lib/Assembly/Writer/Debug:../lib/Analysis/Debug:../lib/VMCore/Debug:../lib/Bytecode/Writer/Debug:../lib/Bytecode/Reader/Debug:../lib/Optimizations/Debug
export LD_LIBRARY_PATH

# Two full cycles are needed for bitwise stability
# FIXME: We must strip symbols, because the symbol tables are not output in 
# sorted order in the bytecode :(

../tools/as/as   < $1      | opt -q -strip > $1.bc.1 || exit 1
../tools/dis/dis < $1.bc.1 > $1.ll.1 || exit 2
../tools/as/as   < $1.ll.1 > $1.bc.2 || exit 3
../tools/dis/dis < $1.bc.2 > $1.ll.2 || exit 4

diff $1.ll.[12] || exit 7

# FIXME: When we sort things correctly and deterministically, we can reenable this
#diff $1.bc.[12] || exit 8

rm $1.[bl][cl].[12]

