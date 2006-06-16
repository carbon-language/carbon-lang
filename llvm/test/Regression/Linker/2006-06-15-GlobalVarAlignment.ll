; The linker should choose the largest alignment when linking.

; RUN: echo "%X = global int 7, align 8" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep 'align 8'

%X = weak global int 7, align 4
