; RUN: llvm-upgrade < %s | llvm-as | llc
; PR1011

	%struct.mng_data = type { sbyte* (%struct.mng_data*, uint)*, int, int, int, sbyte, sbyte, int, int, int, int, int }

implementation   ; Functions:

void %mng_display_bgr565() {
entry:
	br bool false, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	br bool false, label %cond_true48, label %cond_next80

cond_true48:		; preds = %bb.preheader
	%tmp = load ubyte* null		; <ubyte> [#uses=1]
	%tmp51 = cast ubyte %tmp to ushort		; <ushort> [#uses=1]
	%tmp99 = load sbyte* null		; <sbyte> [#uses=1]
	%tmp54 = cast sbyte %tmp99 to ubyte		; <ubyte> [#uses=1]
	%tmp54 = cast ubyte %tmp54 to int		; <int> [#uses=1]
	%tmp55 = lshr int %tmp54, ubyte 3		; <int> [#uses=1]
	%tmp55 = cast int %tmp55 to ushort		; <ushort> [#uses=1]
	%tmp52 = shl ushort %tmp51, ubyte 5		; <ushort> [#uses=1]
	%tmp56 = and ushort %tmp55, 28		; <ushort> [#uses=1]
	%tmp57 = or ushort %tmp56, %tmp52		; <ushort> [#uses=1]
	%tmp60 = cast ushort %tmp57 to uint		; <uint> [#uses=1]
	%tmp62 = xor uint 0, 65535		; <uint> [#uses=1]
	%tmp63 = mul uint %tmp60, %tmp62		; <uint> [#uses=1]
	%tmp65 = add uint 0, %tmp63		; <uint> [#uses=1]
	%tmp69 = add uint 0, %tmp65		; <uint> [#uses=1]
	%tmp70 = lshr uint %tmp69, ubyte 16		; <uint> [#uses=1]
	%tmp70 = cast uint %tmp70 to ushort		; <ushort> [#uses=1]
	%tmp75 = lshr ushort %tmp70, ubyte 8		; <ushort> [#uses=1]
	%tmp75 = cast ushort %tmp75 to ubyte		; <ubyte> [#uses=1]
	%tmp76 = lshr ubyte %tmp75, ubyte 5		; <ubyte> [#uses=1]
	store ubyte %tmp76, ubyte* null
	ret void

cond_next80:		; preds = %bb.preheader
	ret void

return:		; preds = %entry
	ret void
}
