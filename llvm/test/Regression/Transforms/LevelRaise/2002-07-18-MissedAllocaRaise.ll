; Looks like we don't raise alloca's like we do mallocs
;
; RUN: llvm-as < %s | opt -raise | llvm-dis | grep '= cast' | not grep \*

implementation   ; Functions:

int *%X() {
	%reg107 = alloca ubyte, uint 4
	%cast213 = cast ubyte* %reg107 to int*
	ret int* %cast213
}
