; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "implementation linkonce void %foo() { ret void } " | as > Output/%s.2.bc
; RUN: as < %s > Output/%s.1.bc
; RUN: link Output/%s.[12].bc | dis | grep foo | grep linkonce

declare void %foo()
