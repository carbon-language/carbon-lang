; The funcresolve pass will (intentionally) llvm-link an _internal_ function body with an
; external declaration.  Because of this, if we LINK an internal function body into
; a program that already has an external declaration for the function name, we must
; rename the internal function to something that does not conflict.

; RUN: echo "implementation internal int %foo() { ret int 7 }" | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep 'internal' | not grep '%foo('

implementation
declare int %foo() 

int %test() { 
  %X = call int %foo()
  ret int %X
}

