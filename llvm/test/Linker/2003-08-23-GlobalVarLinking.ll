; RUN: llvm-upgrade < %s | llvm-as > %t.out1.bc
; RUN: echo {%S = external global \{ int, opaque* \} declare void %F(opaque*)}\
; RUN:   | llvm-upgrade | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc | llvm-dis | not grep opaque

; After linking this testcase, there should be no opaque types left.  The two
; S's should cause the opaque type to be resolved to 'int'.
%S = global { int, int* } { int 5, int* null }

declare void %F(int*)
