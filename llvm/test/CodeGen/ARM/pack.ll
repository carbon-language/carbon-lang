; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | \
; RUN:   grep pkhbt | wc -l | grep 5
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | \
; RUN:   grep pkhtb | wc -l | grep 4
; END.

implementation   ; Functions:

int %test1(int %X, int %Y) {
	%tmp1 = and int %X, 65535		; <int> [#uses=1]
	%tmp4 = shl int %Y, ubyte 16		; <int> [#uses=1]
	%tmp5 = or int %tmp4, %tmp1		; <int> [#uses=1]
	ret int %tmp5
}

int %test1a(int %X, int %Y) {
	%tmp19 = and int %X, 65535		; <int> [#uses=1]
	%tmp37 = shl int %Y, ubyte 16		; <int> [#uses=1]
	%tmp5 = or int %tmp37, %tmp19		; <int> [#uses=1]
	ret int %tmp5
}

int %test2(int %X, int %Y) {
	%tmp1 = and int %X, 65535		; <int> [#uses=1]
	%tmp3 = shl int %Y, ubyte 12		; <int> [#uses=1]
	%tmp4 = and int %tmp3, -65536		; <int> [#uses=1]
	%tmp57 = or int %tmp4, %tmp1		; <int> [#uses=1]
	ret int %tmp57
}

int %test3(int %X, int %Y) {
	%tmp19 = and int %X, 65535		; <int> [#uses=1]
	%tmp37 = shl int %Y, ubyte 18		; <int> [#uses=1]
	%tmp5 = or int %tmp37, %tmp19		; <int> [#uses=1]
	ret int %tmp5
}

int %test4(int %X, int %Y) {
	%tmp1 = and int %X, 65535		; <int> [#uses=1]
	%tmp3 = and int %Y, -65536		; <int> [#uses=1]
	%tmp46 = or int %tmp3, %tmp1		; <int> [#uses=1]
	ret int %tmp46
}

int %test5(int %X, int %Y) {
	%tmp17 = and int %X, -65536		; <int> [#uses=1]
	%tmp2 = cast int %Y to uint		; <uint> [#uses=1]
	%tmp4 = shr uint %tmp2, ubyte 16		; <uint> [#uses=1]
	%tmp4 = cast uint %tmp4 to int		; <int> [#uses=1]
	%tmp5 = or int %tmp4, %tmp17		; <int> [#uses=1]
	ret int %tmp5
}

int %test5a(int %X, int %Y) {
	%tmp110 = and int %X, -65536		; <int> [#uses=1]
	%Y = cast int %Y to uint		; <uint> [#uses=1]
	%tmp37 = shr uint %Y, ubyte 16		; <uint> [#uses=1]
	%tmp39 = cast uint %tmp37 to int		; <int> [#uses=1]
	%tmp5 = or int %tmp39, %tmp110		; <int> [#uses=1]
	ret int %tmp5
}

int %test6(int %X, int %Y) {
	%tmp1 = and int %X, -65536		; <int> [#uses=1]
	%Y = cast int %Y to uint		; <uint> [#uses=1]
	%tmp37 = shr uint %Y, ubyte 12		; <uint> [#uses=1]
	%tmp38 = cast uint %tmp37 to int		; <int> [#uses=1]
	%tmp4 = and int %tmp38, 65535		; <int> [#uses=1]
	%tmp59 = or int %tmp4, %tmp1		; <int> [#uses=1]
	ret int %tmp59
}

int %test7(int %X, int %Y) {
	%tmp1 = and int %X, -65536		; <int> [#uses=1]
	%tmp3 = shr int %Y, ubyte 18		; <int> [#uses=1]
	%tmp4 = and int %tmp3, 65535		; <int> [#uses=1]
	%tmp57 = or int %tmp4, %tmp1		; <int> [#uses=1]
	ret int %tmp57
}

