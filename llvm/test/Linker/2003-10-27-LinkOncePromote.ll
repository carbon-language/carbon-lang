; The linker should merge link-once globals into strong external globals,
; just like it does for weak symbols!

; RUN: echo "%X = global int 7" | llvm-upgrade | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc

%X = linkonce global int 7
