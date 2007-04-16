; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   grep align.4 | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   grep align.2 | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   grep align.3 | wc -l | grep 1


%A = global <4 x uint> < uint 10, uint 20, uint 30, uint 40 >
%B = global float 1.000000e+02
%C = global double 2.000000e+03

