; RUN: opt < %s -scalarrepl -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-apple-darwin10.0.0"

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


define i64 @test6(<2 x float> %X) {
	%X_addr = alloca <2 x float>
        store <2 x float> %X, <2 x float>* %X_addr
	%P = bitcast <2 x float>* %X_addr to i64*
	%tmp = load i64* %P
	ret i64 %tmp
; CHECK: @test6
; CHECK: bitcast <2 x float> %X to <1 x i64>
; CHECK: ret i64
}

define float @test7(<4 x float> %x) {
	%a = alloca <4 x float>
	store <4 x float> %x, <4 x float>* %a
	%p = bitcast <4 x float>* %a to <2 x float>*
	%b = load <2 x float>* %p
	%q = getelementptr <4 x float>* %a, i32 0, i32 2
	%c = load float* %q
	ret float %c
; CHECK: @test7
; CHECK-NOT: alloca
; CHECK: bitcast <4 x float> %x to <2 x double>
; CHECK-NEXT: extractelement <2 x double>
; CHECK-NEXT: bitcast double %tmp4 to <2 x float>
; CHECK-NEXT: extractelement <4 x float>
}

define void @test8(<4 x float> %x, <2 x float> %y) {
	%a = alloca <4 x float>
	store <4 x float> %x, <4 x float>* %a
	%p = bitcast <4 x float>* %a to <2 x float>*
	store <2 x float> %y, <2 x float>* %p
	ret void
; CHECK: @test8
; CHECK-NOT: alloca
; CHECK: bitcast <4 x float> %x to <2 x double>
; CHECK-NEXT: bitcast <2 x float> %y to double
; CHECK-NEXT: insertelement <2 x double>
; CHECK-NEXT: bitcast <2 x double> %tmp2 to <4 x float>
}

define i256 @test9(<4 x i256> %x) {
	%a = alloca <4 x i256>
	store <4 x i256> %x, <4 x i256>* %a
	%p = bitcast <4 x i256>* %a to <2 x i256>*
	%b = load <2 x i256>* %p
	%q = getelementptr <4 x i256>* %a, i32 0, i32 2
	%c = load i256* %q
	ret i256 %c
; CHECK: @test9
; CHECK-NOT: alloca
; CHECK: bitcast <4 x i256> %x to <2 x i512>
; CHECK-NEXT: extractelement <2 x i512>
; CHECK-NEXT: bitcast i512 %tmp4 to <2 x i256>
; CHECK-NEXT: extractelement <4 x i256>
}

define void @test10(<4 x i256> %x, <2 x i256> %y) {
	%a = alloca <4 x i256>
	store <4 x i256> %x, <4 x i256>* %a
	%p = bitcast <4 x i256>* %a to <2 x i256>*
	store <2 x i256> %y, <2 x i256>* %p
	ret void
; CHECK: @test10
; CHECK-NOT: alloca
; CHECK: bitcast <4 x i256> %x to <2 x i512>
; CHECK-NEXT: bitcast <2 x i256> %y to i512
; CHECK-NEXT: insertelement <2 x i512>
; CHECK-NEXT: bitcast <2 x i512> %tmp2 to <4 x i256>
}

%union.v = type { <2 x i64> }

define void @test11(<2 x i64> %x) {
  %a = alloca %union.v
  %p = getelementptr inbounds %union.v* %a, i32 0, i32 0
  store <2 x i64> %x, <2 x i64>* %p, align 16
  %q = getelementptr inbounds %union.v* %a, i32 0, i32 0
  %r = bitcast <2 x i64>* %q to <4 x float>*
  %b = load <4 x float>* %r, align 16
  ret void
; CHECK: @test11
; CHECK-NOT: alloca
}
