; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "%X = uninitialized global [1x int]" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep constant

%X = constant [1 x int] [ int 12 ]
