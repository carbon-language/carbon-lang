; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < rlwimi2.ll | llc -march=ppc32 | grep rlwimi | wc -l | grep 3 &&
; RUN: llvm-as < rlwimi2.ll | llc -march=ppc32 | grep srwi   | wc -l | grep 1 &&
; RUN: llvm-as < rlwimi2.ll | llc -march=ppc32 | not grep slwi

implementation   ; Functions:

ushort %test1(uint %srcA, uint %srcB, uint %alpha) {
entry:
	%tmp.1 = shl uint %srcA, ubyte 15		; <uint> [#uses=1]
	%tmp.4 = and uint %tmp.1, 32505856		; <uint> [#uses=1]
	%tmp.6 = and uint %srcA, 31775		; <uint> [#uses=1]
	%tmp.7 = or uint %tmp.4, %tmp.6		; <uint> [#uses=1]
	%tmp.9 = shl uint %srcB, ubyte 15		; <uint> [#uses=1]
	%tmp.12 = and uint %tmp.9, 32505856		; <uint> [#uses=1]
	%tmp.14 = and uint %srcB, 31775		; <uint> [#uses=1]
	%tmp.15 = or uint %tmp.12, %tmp.14		; <uint> [#uses=1]
	%tmp.18 = mul uint %tmp.7, %alpha		; <uint> [#uses=1]
	%tmp.20 = sub uint 32, %alpha		; <uint> [#uses=1]
	%tmp.22 = mul uint %tmp.15, %tmp.20		; <uint> [#uses=1]
	%tmp.23 = add uint %tmp.22, %tmp.18		; <uint> [#uses=2]
	%tmp.27 = shr uint %tmp.23, ubyte 5		; <uint> [#uses=1]
	%tmp.28 = cast uint %tmp.27 to ushort		; <ushort> [#uses=1]
	%tmp.29 = and ushort %tmp.28, 31775		; <ushort> [#uses=1]
	%tmp.33 = shr uint %tmp.23, ubyte 20		; <uint> [#uses=1]
	%tmp.34 = cast uint %tmp.33 to ushort		; <ushort> [#uses=1]
	%tmp.35 = and ushort %tmp.34, 992		; <ushort> [#uses=1]
	%tmp.36 = or ushort %tmp.29, %tmp.35		; <ushort> [#uses=1]
	ret ushort %tmp.36
}
