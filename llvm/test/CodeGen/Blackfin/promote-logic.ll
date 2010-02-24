; RUN: llc < %s -march=bfin 

; DAGCombiner::SimplifyBinOpWithSameOpcodeHands can produce an illegal i16 OR
; operation after LegalizeOps.

define void @mng_display_bgr565() {
entry:
	br i1 false, label %bb.preheader, label %return

bb.preheader:
	br i1 false, label %cond_true48, label %cond_next80

cond_true48:
	%tmp = load i8* null
	%tmp51 = zext i8 %tmp to i16
	%tmp99 = load i8* null
	%tmp54 = bitcast i8 %tmp99 to i8
	%tmp54.upgrd.1 = zext i8 %tmp54 to i32
	%tmp55 = lshr i32 %tmp54.upgrd.1, 3
	%tmp55.upgrd.2 = trunc i32 %tmp55 to i16
	%tmp52 = shl i16 %tmp51, 5
	%tmp56 = and i16 %tmp55.upgrd.2, 28
	%tmp57 = or i16 %tmp56, %tmp52
	%tmp60 = zext i16 %tmp57 to i32
	%tmp62 = xor i32 0, 65535
	%tmp63 = mul i32 %tmp60, %tmp62
	%tmp65 = add i32 0, %tmp63
	%tmp69 = add i32 0, %tmp65
	%tmp70 = lshr i32 %tmp69, 16
	%tmp70.upgrd.3 = trunc i32 %tmp70 to i16
	%tmp75 = lshr i16 %tmp70.upgrd.3, 8
	%tmp75.upgrd.4 = trunc i16 %tmp75 to i8
	%tmp76 = lshr i8 %tmp75.upgrd.4, 5
	store i8 %tmp76, i8* null
	ret void

cond_next80:
	ret void

return:
	ret void
}
