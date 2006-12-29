; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "implementation linkonce void %foo() { ret void } " | llvm-upgrade|\
; RUN:    llvm-as -o %t.2.bc -f
; RUN: llvm-upgrade < %s | llvm-as -o %t.1.bc -f
; RUN: llvm-link %t.[12].bc | llvm-dis | grep foo | grep linkonce

declare void %foo()
