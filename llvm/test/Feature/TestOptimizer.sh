#!/bin/sh
LD_LIBRARY_PATH=../lib/Assembly/Parser/Debug:../lib/Assembly/Writer/Debug:../lib/Analysis/Debug:../lib/VMCore/Debug:../lib/Bytecode/Writer/Debug:../lib/Bytecode/Reader/Debug:../lib/Optimizations/Debug
export LD_LIBRARY_PATH


../tools/as/as < $1 | ../tools/opt/opt -q -inline -dce -constprop -dce | ../tools/dis/dis | ../tools/as/as > $1.bc.1 || exit 1

# Should not be able to optimize further!
../tools/opt/opt -q -constprop -dce < $1.bc.1 > $1.bc.2 || exit 2

../tools/dis/dis < $1.bc.1 > $1.ll.1 || exit 3
../tools/dis/dis < $1.bc.2 > $1.ll.2 || exit 3
diff $1.ll.[12] || exit 3

# Try out SCCP
../tools/as/as < $1 | ../tools/opt/opt -q -inline -dce -sccp -dce | ../tools/dis/dis | ../tools/as/as > $1.bc.3 || exit 1

# Should not be able to optimize further!
#../tools/opt/opt -q -sccp -dce < $1.bc.3 > $1.bc.4 || exit 2

#diff $1.bc.[34] || exit 3
rm $1.bc.[123] $1.ll.[12]

