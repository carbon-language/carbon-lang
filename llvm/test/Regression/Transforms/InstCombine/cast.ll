; Tests to make sure elimination of casts is working correctly

; RUN: if as < %s | opt -instcombine -die | dis | grep '%c' | grep cast
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int %test1(int %A) {
	%c1 = cast int %A to uint
	%c2 = cast uint %c1 to int
	ret int %c2
}

ulong %test2(ubyte %A) {
	%c1 = cast ubyte %A to ushort
	%c2 = cast ushort %c1 to uint
	%Ret = cast uint %c2 to ulong
	ret ulong %Ret
}

ulong %test3(ulong %A) {    ; This function should just use bitwise AND
	%c1 = cast ulong %A to ubyte
	%c2 = cast ubyte %c1 to ulong
	ret ulong %c2
}
