; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "%Ty = type opaque" | as > Output/%s.1.bc
; RUN: as < %s > Output/%s.2.bc
; RUN: link Output/%s.[12].bc | dis | grep '%Ty ' | grep -v opaque

%Ty = type int

