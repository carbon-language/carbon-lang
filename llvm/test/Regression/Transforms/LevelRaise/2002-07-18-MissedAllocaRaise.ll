; Looks like we don't raise alloca's like we do mallocs
;
; RUN: if as < %s | opt -raise | dis | grep '= cast' | grep \*
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation   ; Functions:

int *%X() {
	%reg107 = alloca ubyte, uint 4
	%cast213 = cast ubyte* %reg107 to int*
	ret int* %cast213
}
