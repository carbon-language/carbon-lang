; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "implementation linkonce void %foo() { ret void } " | as > %t.2.bc
; RUN: as < %s > %t.1.bc
; RUN: link %t.[12].bc | dis | grep foo | grep linkonce

declare void %foo()
