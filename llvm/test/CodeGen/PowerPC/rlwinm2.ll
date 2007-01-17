; All of these ands and shifts should be folded into rlw[i]nm instructions
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep and && 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep srawi && 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep srwi && 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep slwi && 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep rlwnm | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep rlwinm | wc -l | grep 1


implementation   ; Functions:

uint %test1(uint %X, int %Y) {
entry:
	%tmp = cast int %Y to ubyte		; <ubyte> [#uses=2]
	%tmp1 = shl uint %X, ubyte %tmp		; <uint> [#uses=1]
	%tmp2 = sub ubyte 32, %tmp		; <ubyte> [#uses=1]
	%tmp3 = shr uint %X, ubyte %tmp2		; <uint> [#uses=1]
	%tmp4 = or uint %tmp1, %tmp3		; <uint> [#uses=1]
	%tmp6 = and uint %tmp4, 127		; <uint> [#uses=1]
	ret uint %tmp6
}

uint %test2(uint %X) {
entry:
	%tmp1 = shr uint %X, ubyte 27		; <uint> [#uses=1]
	%tmp2 = shl uint %X, ubyte 5		; <uint> [#uses=1]
	%tmp2.masked = and uint %tmp2, 96		; <uint> [#uses=1]
	%tmp5 = or uint %tmp1, %tmp2.masked		; <uint> [#uses=1]
	ret uint %tmp5
}
