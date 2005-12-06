; RUN: echo "%G = appending global [0 x int] zeroinitializer" | llvm-as > %t.out2.bc
; RUN: llvm-as < %s > %t.out1.bc
; RUN: llvm-link %t.out[12].bc | llvm-dis | grep '%G ='

; When linked, the globals should be merged, and the result should still 
; be named '%G'.

%G = appending global [1 x int] zeroinitializer
