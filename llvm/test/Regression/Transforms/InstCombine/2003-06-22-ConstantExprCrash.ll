; This is a bug in the VMcode library, not instcombine, it's just convenient 
; to expose it here.

; RUN: as < %s | opt -instcombine -disable-output

%A = global int 1
%B = global int 2

bool %test() {
	%C = setlt int* getelementptr (int* %A, long 1), getelementptr (int* %B, long 2)    ; Will get promoted to constantexpr
	ret bool %C
}
