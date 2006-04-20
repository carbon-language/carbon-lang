; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5

void %test(sbyte* %stack) {
entry:
	%tmp9 = seteq int 0, 0		; <bool> [#uses=1]
	%tmp30 = seteq uint 0, 0		; <bool> [#uses=1]
	br bool %tmp30, label %cond_next54, label %cond_true31

cond_true860:		; preds = %bb855
	%tmp879 = tail call <4 x float> %llvm.ppc.altivec.vmaddfp( <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp880 = cast <4 x float> %tmp879 to <4 x int>		; <<4 x int>> [#uses=2]
	%tmp883 = shufflevector <4 x int> %tmp880, <4 x int> undef, <4 x uint> < uint 1, uint 1, uint 1, uint 1 >		; <<4 x int>> [#uses=1]
	%tmp883 = cast <4 x int> %tmp883 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp885 = shufflevector <4 x int> %tmp880, <4 x int> undef, <4 x uint> < uint 2, uint 2, uint 2, uint 2 >		; <<4 x int>> [#uses=1]
	%tmp885 = cast <4 x int> %tmp885 to <4 x float>		; <<4 x float>> [#uses=1]
	br label %cond_next905

cond_true31:		; preds = %entry
	ret void

cond_next54:		; preds = %entry
	br bool %tmp9, label %cond_false385, label %bb279

bb279:		; preds = %cond_next54
	ret void

cond_false385:		; preds = %cond_next54
	%tmp388 = seteq uint 0, 0		; <bool> [#uses=1]
	br bool %tmp388, label %cond_next463, label %cond_true389

cond_true389:		; preds = %cond_false385
	ret void

cond_next463:		; preds = %cond_false385
	%tmp1208107 = setgt sbyte* null, %stack		; <bool> [#uses=1]
	br bool %tmp1208107, label %cond_true1209.preheader, label %bb1212

cond_true498:		; preds = %cond_true1209.preheader
	ret void

cond_true519:		; preds = %cond_true1209.preheader
	%bothcond = or bool false, false		; <bool> [#uses=1]
	br bool %bothcond, label %bb855, label %bb980

cond_false548:		; preds = %cond_true1209.preheader
	ret void

bb855:		; preds = %cond_true519
	%tmp859 = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp859, label %cond_true860, label %cond_next905

cond_next905:		; preds = %bb855, %cond_true860
	%vfpw2.4 = phi <4 x float> [ %tmp885, %cond_true860 ], [ undef, %bb855 ]		; <<4 x float>> [#uses=0]
	%vfpw1.4 = phi <4 x float> [ %tmp883, %cond_true860 ], [ undef, %bb855 ]		; <<4 x float>> [#uses=0]
	%tmp930 = cast <4 x float> zeroinitializer to <4 x int>		; <<4 x int>> [#uses=0]
	ret void

bb980:		; preds = %cond_true519
	ret void

cond_true1209.preheader:		; preds = %cond_next463
	%tmp496 = and uint 0, 12288		; <uint> [#uses=1]
	switch uint %tmp496, label %cond_false548 [
		 uint 0, label %cond_true498
		 uint 4096, label %cond_true519
	]

bb1212:		; preds = %cond_next463
	ret void
}

declare <4 x float> %llvm.ppc.altivec.vmaddfp(<4 x float>, <4 x float>, <4 x float>)
