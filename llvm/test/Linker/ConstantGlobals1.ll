; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo {%X = constant \[1 x int\] \[int 8\] } | \
; RUN:   llvm-upgrade | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc | llvm-dis | grep constant

%X = uninitialized global [1 x int]
