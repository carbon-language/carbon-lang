; RUN: echo "%G = weak global {{{{double}}}} zeroinitializer" | llvm-as > %t.out2.bc
; RUN: llvm-as < %s > %t.out1.bc
; RUN: llvm-link %t.out[12].bc | llvm-dis | not grep '\}'

; When linked, the global above should be eliminated, being merged with the 
; global below.

%G = global double 1.0
