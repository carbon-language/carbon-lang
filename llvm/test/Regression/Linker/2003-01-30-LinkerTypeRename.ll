; This fails because the linker renames the non-opaque type not the opaque 
; one...

; RUN: echo "%Ty = type opaque" | as > %t.1.bc
; RUN: as < %s > %t.2.bc
; RUN: link %t.[12].bc | dis | grep '%Ty ' | grep -v opaque

%Ty = type int

