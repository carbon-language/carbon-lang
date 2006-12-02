; Make sure that the ds-opt pass is constantizing globals
;
; RUN: llvm-upgrade < %s | llvm-as | opt -ds-opt | llvm-dis | grep %G | grep constant


%G = internal global int 0		; <int*> [#uses=2]

implementation   ; Functions:

int %main() {
	%A = load int* %G
	ret int %A
}
