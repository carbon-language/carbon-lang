; The funcresolve pass will (intentionally) link an _internal_ function body with an
; external declaration.  Because of this, if we LINK an internal function body into
; a program that already has an external declaration for the function name, we must
; rename the internal function to something that does not conflict.

; RUN: echo "implementation internal int %foo() { ret int 7 }" | as > %t.1.bc
; RUN: as < %s > %t.2.bc
; RUN: link %t.[12].bc | dis | grep 'internal' | not grep '%foo('

implementation
declare int %foo() 

int %test() { 
  %X = call int %foo()
  ret int %X
}

