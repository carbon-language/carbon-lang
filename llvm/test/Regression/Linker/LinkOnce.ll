; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "%X = linkonce global int 8" | as > Output/%s.2.bc
; RUN: as < %s > Output/%s.1.bc
; RUN: link Output/%s.[12].bc | dis

%X = linkonce global int 7
