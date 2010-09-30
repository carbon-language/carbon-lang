; RUN: llc < %s -march=x86 -mattr=+mmx | grep pextrd
; RUN: llc < %s -march=x86 -mattr=+mmx | grep punpckhdq | count 1
; There are no MMX operations in bork; promoted to XMM.

define void @bork(<1 x i64>* %x) {
entry:
	%tmp2 = load <1 x i64>* %x		; <<1 x i64>> [#uses=1]
	%tmp6 = bitcast <1 x i64> %tmp2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	%tmp9 = shufflevector <2 x i32> %tmp6, <2 x i32> undef, <2 x i32> < i32 1, i32 1 >		; <<2 x i32>> [#uses=1]
	%tmp10 = bitcast <2 x i32> %tmp9 to <1 x i64>		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp10, <1 x i64>* %x
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

; pork uses MMX.

define void @pork(x86_mmx* %x) {
entry:
	%tmp2 = load x86_mmx* %x		; <x86_mmx> [#uses=1]
        %tmp9 = tail call x86_mmx @llvm.x86.mmx.punpckhdq (x86_mmx %tmp2, x86_mmx %tmp2)
	store x86_mmx %tmp9, x86_mmx* %x
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare x86_mmx @llvm.x86.mmx.punpckhdq(x86_mmx, x86_mmx)
declare void @llvm.x86.mmx.emms()
