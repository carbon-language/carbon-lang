; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "%X = constant [1x int] [int 8]" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep constant

%X = uninitialized global [1 x int]
