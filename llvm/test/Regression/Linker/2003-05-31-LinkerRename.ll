; The funcresolve pass will (intentionally) link an _internal_ function body with an
; external declaration.  Because of this, if we LINK an internal function body into
; a program that already has an external declaration for the function name, we must
; rename the internal function to something that does not conflict.

; RUN: echo "implementation internal int %foo() { ret int 7 }" | as > Output/%s.1.bc
; RUN: as < %s > Output/%s.2.bc
; RUN: if link Output/%s.[12].bc | dis | grep 'internal' | grep '%foo('
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation
declare int %foo() 

int %test() { 
  %X = call int %foo()
  ret int %X
}

