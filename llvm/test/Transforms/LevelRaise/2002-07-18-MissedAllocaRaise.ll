; Looks like we don't raise alloca's like we do mallocs
; XFAIL: *
; RUN: llvm-upgrade < %s | llvm-as | opt -raise | llvm-dis | not grep bitcast

implementation   ; Functions:

int *%X() {
	%reg107 = alloca ubyte, uint 4
	%cast213 = cast ubyte* %reg107 to int*
	ret int* %cast213
}
