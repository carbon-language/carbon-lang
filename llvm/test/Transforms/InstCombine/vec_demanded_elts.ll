; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {sub float}
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {mul float}
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep {insertelement.*0.00}
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep {call.*llvm.x86.sse.mul}
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep {call.*llvm.x86.sse.sub}
; END.

ushort %Convert_sse(float %f) {
entry:
	%tmp = insertelement <4 x float> undef, float %f, uint 0		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, uint 2		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, uint 3		; <<4 x float>> [#uses=1]
	%tmp28 = tail call <4 x float> %llvm.x86.sse.sub.ss( <4 x float> %tmp12, <4 x float> < float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp37 = tail call <4 x float> %llvm.x86.sse.mul.ss( <4 x float> %tmp28, <4 x float> < float 5.000000e-01, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> %llvm.x86.sse.min.ss( <4 x float> %tmp37, <4 x float> < float 6.553500e+04, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> %llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp = tail call int %llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <int> [#uses=1]
	%tmp69 = cast int %tmp to ushort		; <ushort> [#uses=1]
	ret ushort %tmp69
}

declare <4 x float> %llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

declare <4 x float> %llvm.x86.sse.mul.ss(<4 x float>, <4 x float>)

declare <4 x float> %llvm.x86.sse.min.ss(<4 x float>, <4 x float>)

declare <4 x float> %llvm.x86.sse.max.ss(<4 x float>, <4 x float>)

declare int %llvm.x86.sse.cvttss2si(<4 x float>)
