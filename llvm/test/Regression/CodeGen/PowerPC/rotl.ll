; RUN: llvm-as < %s | llc -march=ppc32 | not grep or && 
; RUN: llvm-as < %s | llc -march=ppc32 | grep rlwnm | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep rlwinm | wc -l | grep 2

implementation   ; Functions:

int %rotlw(uint %x, int %sh) {
entry:
	%tmp.3 = cast int %sh to ubyte		; <ubyte> [#uses=1]
	%x = cast uint %x to int		; <int> [#uses=1]
	%tmp.7 = sub int 32, %sh		; <int> [#uses=1]
	%tmp.9 = cast int %tmp.7 to ubyte		; <ubyte> [#uses=1]
	%tmp.10 = shr uint %x, ubyte %tmp.9		; <uint> [#uses=1]
	%tmp.4 = shl int %x, ubyte %tmp.3		; <int> [#uses=1]
	%tmp.10 = cast uint %tmp.10 to int		; <int> [#uses=1]
	%tmp.12 = or int %tmp.10, %tmp.4		; <int> [#uses=1]
	ret int %tmp.12
}

int %rotrw(uint %x, int %sh) {
entry:
	%tmp.3 = cast int %sh to ubyte		; <ubyte> [#uses=1]
	%tmp.4 = shr uint %x, ubyte %tmp.3		; <uint> [#uses=1]
	%tmp.7 = sub int 32, %sh		; <int> [#uses=1]
	%tmp.9 = cast int %tmp.7 to ubyte		; <ubyte> [#uses=1]
	%x = cast uint %x to int		; <int> [#uses=1]
	%tmp.4 = cast uint %tmp.4 to int		; <int> [#uses=1]
	%tmp.10 = shl int %x, ubyte %tmp.9		; <int> [#uses=1]
	%tmp.12 = or int %tmp.4, %tmp.10		; <int> [#uses=1]
	ret int %tmp.12
}

int %rotlwi(uint %x) {
entry:
	%x = cast uint %x to int		; <int> [#uses=1]
	%tmp.7 = shr uint %x, ubyte 27		; <uint> [#uses=1]
	%tmp.3 = shl int %x, ubyte 5		; <int> [#uses=1]
	%tmp.7 = cast uint %tmp.7 to int		; <int> [#uses=1]
	%tmp.9 = or int %tmp.3, %tmp.7		; <int> [#uses=1]
	ret int %tmp.9
}

int %rotrwi(uint %x) {
entry:
	%tmp.3 = shr uint %x, ubyte 5		; <uint> [#uses=1]
	%x = cast uint %x to int		; <int> [#uses=1]
	%tmp.3 = cast uint %tmp.3 to int		; <int> [#uses=1]
	%tmp.7 = shl int %x, ubyte 27		; <int> [#uses=1]
	%tmp.9 = or int %tmp.3, %tmp.7		; <int> [#uses=1]
	ret int %tmp.9
}
