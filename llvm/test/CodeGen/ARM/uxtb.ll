; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | \
; RUN:   grep uxt | wc -l | grep 10
; END.

uint %test1(uint %x) {
	%tmp1 = and uint %x, 16711935		; <uint> [#uses=1]
	ret uint %tmp1
}

uint %test2(uint %x) {
	%tmp1 = shr uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 16711935		; <uint> [#uses=1]
	ret uint %tmp2
}

uint %test3(uint %x) {
	%tmp1 = shr uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 16711935		; <uint> [#uses=1]
	ret uint %tmp2
}

uint %test4(uint %x) {
	%tmp1 = shr uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp6 = and uint %tmp1, 16711935		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test5(uint %x) {
	%tmp1 = shr uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 16711935		; <uint> [#uses=1]
	ret uint %tmp2
}

uint %test6(uint %x) {
	%tmp1 = shr uint %x, ubyte 16		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 255		; <uint> [#uses=1]
	%tmp4 = shl uint %x, ubyte 16		; <uint> [#uses=1]
	%tmp5 = and uint %tmp4, 16711680		; <uint> [#uses=1]
	%tmp6 = or uint %tmp2, %tmp5		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test7(uint %x) {
	%tmp1 = shr uint %x, ubyte 16		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 255		; <uint> [#uses=1]
	%tmp4 = shl uint %x, ubyte 16		; <uint> [#uses=1]
	%tmp5 = and uint %tmp4, 16711680		; <uint> [#uses=1]
	%tmp6 = or uint %tmp2, %tmp5		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test8(uint %x) {
	%tmp1 = shl uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp2 = and uint %tmp1, 16711680		; <uint> [#uses=1]
	%tmp5 = shr uint %x, ubyte 24		; <uint> [#uses=1]
	%tmp6 = or uint %tmp2, %tmp5		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test9(uint %x) {
	%tmp1 = shr uint %x, ubyte 24		; <uint> [#uses=1]
	%tmp4 = shl uint %x, ubyte 8		; <uint> [#uses=1]
	%tmp5 = and uint %tmp4, 16711680		; <uint> [#uses=1]
	%tmp6 = or uint %tmp5, %tmp1		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test10(uint %p0) {
        %tmp1 = shr uint %p0, ubyte 7           ; <uint> [#uses=1]
        %tmp2 = and uint %tmp1, 16253176                ; <uint> [#uses=2]
        %tmp4 = shr uint %tmp2, ubyte 5         ; <uint> [#uses=1]
        %tmp5 = and uint %tmp4, 458759          ; <uint> [#uses=1]
        %tmp7 = or uint %tmp5, %tmp2            ; <uint> [#uses=1]
        ret uint %tmp7
}

