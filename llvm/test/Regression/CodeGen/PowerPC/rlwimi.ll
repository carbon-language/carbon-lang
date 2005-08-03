; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < rlwimi.ll | llc -march=ppc32 | not grep and && 
; RUN: llvm-as < rlwimi.ll | llc -march=ppc32 | grep rlwimi | wc -l | grep 8

implementation   ; Functions:

int %test1(int %x, int %y) {
entry:
	%tmp.3 = shl int %x, ubyte 16		; <int> [#uses=1]
	%tmp.7 = and int %y, 65535		; <int> [#uses=1]
	%tmp.9 = or int %tmp.7, %tmp.3		; <int> [#uses=1]
	ret int %tmp.9
}

int %test2(int %x, int %y) {
entry:
	%tmp.7 = and int %x, 65535		; <int> [#uses=1]
	%tmp.3 = shl int %y, ubyte 16		; <int> [#uses=1]
	%tmp.9 = or int %tmp.7, %tmp.3		; <int> [#uses=1]
	ret int %tmp.9
}

uint %test3(uint %x, uint %y) {
entry:
	%tmp.3 = shr uint %x, ubyte 16		; <uint> [#uses=1]
	%tmp.6 = and uint %y, 4294901760		; <uint> [#uses=1]
	%tmp.7 = or uint %tmp.6, %tmp.3		; <uint> [#uses=1]
	ret uint %tmp.7
}

uint %test4(uint %x, uint %y) {
entry:
	%tmp.6 = and uint %x, 4294901760		; <uint> [#uses=1]
	%tmp.3 = shr uint %y, ubyte 16		; <uint> [#uses=1]
	%tmp.7 = or uint %tmp.6, %tmp.3		; <uint> [#uses=1]
	ret uint %tmp.7
}

int %test5(int %x, int %y) {
entry:
	%tmp.3 = shl int %x, ubyte 1		; <int> [#uses=1]
	%tmp.4 = and int %tmp.3, -65536		; <int> [#uses=1]
	%tmp.7 = and int %y, 65535		; <int> [#uses=1]
	%tmp.9 = or int %tmp.4, %tmp.7		; <int> [#uses=1]
	ret int %tmp.9
}

int %test6(int %x, int %y) {
entry:
	%tmp.7 = and int %x, 65535		; <int> [#uses=1]
	%tmp.3 = shl int %y, ubyte 1		; <int> [#uses=1]
	%tmp.4 = and int %tmp.3, -65536		; <int> [#uses=1]
	%tmp.9 = or int %tmp.4, %tmp.7		; <int> [#uses=1]
	ret int %tmp.9
}

int %test7(int %x, int %y) {
entry:
	%tmp.2 = and int %x, -65536		; <int> [#uses=1]
	%tmp.5 = and int %y, 65535		; <int> [#uses=1]
	%tmp.7 = or int %tmp.5, %tmp.2		; <int> [#uses=1]
	ret int %tmp.7
}

uint %test8(uint %bar) {
entry:
	%tmp.3 = shl uint %bar, ubyte 1		; <uint> [#uses=1]
	%tmp.4 = and uint %tmp.3, 2		; <uint> [#uses=1]
	%tmp.6 = and uint %bar, 4294967293		; <uint> [#uses=1]
	%tmp.7 = or uint %tmp.4, %tmp.6		; <uint> [#uses=1]
	ret uint %tmp.7
}
