; The linker should merge link-once globals into strong external globals,
; just like it does for weak symbols!

; RUN: echo "%X = global int 7" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc

%X = linkonce global int 7
