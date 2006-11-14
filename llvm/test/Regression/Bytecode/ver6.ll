; Just see if we can disassemble the ver6.ll.bc bc file for upgrade purposes.
; RUN: llvm-dis < %s.bc | llvm-as | llc -o /dev/null -f -march=c
; RUN: llvm-dis < %s.bc | llvm-as | opt -verify | llvm-dis | llvm-as -o /dev/null -f
