; This fails linking when it is linked with an empty file as the first object file

; RUN: llvm-as > %t1.bc < /dev/null
; RUN: llvm-upgrade < %s | llvm-as > %t2.bc
; RUN: llvm-link %t1.bc %t2.bc

%work = global int (int, int)* %zip

declare int %zip(int, int)
