; This fails because the linker renames the external symbol not the internal 
; one...

; RUN: echo "implementation internal int %foo() { ret int 7 }" | as > Output/%s.1.bc
; RUN: as < %s > Output/%s.2.bc
; RUN: link Output/%s.[12].bc | dis | grep '%foo()' | grep -v internal

implementation
int %foo() { ret int 0 }

