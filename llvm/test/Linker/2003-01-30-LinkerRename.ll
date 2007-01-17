; This fails because the linker renames the external symbol not the internal 
; one...

; RUN: echo "implementation internal int %foo() { ret int 7 }" | llvm-upgrade |\
; RUN:    llvm-as > %t.1.bc
; RUN: llvm-upgrade < %s | llvm-as -o %t.2.bc -f
; RUN: llvm-link %t.[12].bc | llvm-dis | grep '%foo()' | grep -v internal

implementation
int %foo() { ret int 0 }

