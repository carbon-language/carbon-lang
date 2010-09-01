; RUN: opt < %s -scalarrepl -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define void @test1(<4 x float>* %F, float %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=3]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%G.upgrd.1 = getelementptr <4 x float>* %G, i32 0, i32 0		; <float*> [#uses=1]
	store float %f, float* %G.upgrd.1
	%tmp4 = load <4 x float>* %G		; <<4 x float>> [#uses=2]
	%tmp6 = fadd <4 x float> %tmp4, %tmp4		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
; CHECK: @test1
; CHECK-NOT: alloca
; CHECK: %tmp = load <4 x float>* %F
; CHECK: fadd <4 x float> %tmp, %tmp
; CHECK-NEXT: insertelement <4 x float> %tmp3, float %f, i32 0
}

define void @test2(<4 x float>* %F, float %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=3]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp.upgrd.2 = getelementptr <4 x float>* %G, i32 0, i32 2		; <float*> [#uses=1]
	store float %f, float* %tmp.upgrd.2
	%tmp4 = load <4 x float>* %G		; <<4 x float>> [#uses=2]
	%tmp6 = fadd <4 x float> %tmp4, %tmp4		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
; CHECK: @test2
; CHECK-NOT: alloca
; CHECK: %tmp = load <4 x float>* %F
; CHECK: fadd <4 x float> %tmp, %tmp
; CHECK-NEXT: insertelement <4 x float> %tmp3, float %f, i32 2
}

define void @test3(<4 x float>* %F, float* %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=2]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp.upgrd.3 = getelementptr <4 x float>* %G, i32 0, i32 2		; <float*> [#uses=1]
	%tmp.upgrd.4 = load float* %tmp.upgrd.3		; <float> [#uses=1]
	store float %tmp.upgrd.4, float* %f
	ret void
; CHECK: @test3
; CHECK-NOT: alloca
; CHECK: %tmp = load <4 x float>* %F
; CHECK: fadd <4 x float> %tmp, %tmp
; CHECK-NEXT: extractelement <4 x float> %tmp3, i32 2
}

define void @test4(<4 x float>* %F, float* %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=2]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%G.upgrd.5 = getelementptr <4 x float>* %G, i32 0, i32 0		; <float*> [#uses=1]
	%tmp.upgrd.6 = load float* %G.upgrd.5		; <float> [#uses=1]
	store float %tmp.upgrd.6, float* %f
	ret void
; CHECK: @test4
; CHECK-NOT: alloca
; CHECK: %tmp = load <4 x float>* %F
; CHECK: fadd <4 x float> %tmp, %tmp
; CHECK-NEXT: extractelement <4 x float> %tmp3, i32 0
}

define i32 @test5(float %X) {  ;; should turn into bitcast.
	%X_addr = alloca [4 x float]
        %X1 = getelementptr [4 x float]* %X_addr, i32 0, i32 2
	store float %X, float* %X1
	%a = bitcast float* %X1 to i32*
	%tmp = load i32* %a
	ret i32 %tmp
; CHECK: @test5
; CHECK-NEXT: bitcast float %X to i32
; CHECK-NEXT: ret i32
}

