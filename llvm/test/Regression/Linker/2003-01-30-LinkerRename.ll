; This fails because the linker renames the external symbol not the internal 
; one...

; RUN: echo "implementation internal int %foo() { ret int 7 }" | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep '%foo()' | grep -v internal

implementation
int %foo() { ret int 0 }

