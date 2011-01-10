; RUN: opt < %s -instcombine -S | FileCheck %s

define i16 @test1(float %f) {
entry:
; CHECK: @test1
; CHECK: fmul float
; CHECK-NOT: insertelement {{.*}} 0.00
; CHECK-NOT: call {{.*}} @llvm.x86.sse.mul
; CHECK-NOT: call {{.*}} @llvm.x86.sse.sub
; CHECK: ret
	%tmp = insertelement <4 x float> undef, float %f, i32 0		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp28 = tail call <4 x float> @llvm.x86.sse.sub.ss( <4 x float> %tmp12, <4 x float> < float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp37 = tail call <4 x float> @llvm.x86.sse.mul.ss( <4 x float> %tmp28, <4 x float> < float 5.000000e-01, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp37, <4 x float> < float 6.553500e+04, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> @llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = tail call i32 @llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <i32> [#uses=1]
	%tmp69 = trunc i32 %tmp.upgrd.1 to i16		; <i16> [#uses=1]
	ret i16 %tmp69
}

define i32 @test2(float %f) {
; CHECK: @test2
; CHECK-NOT: insertelement
; CHECK-NOT: extractelement
; CHECK: ret
  %tmp5 = fmul float %f, %f
  %tmp9 = insertelement <4 x float> undef, float %tmp5, i32 0
  %tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 1
  %tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2
  %tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3
  %tmp19 = bitcast <4 x float> %tmp12 to <4 x i32>
  %tmp21 = extractelement <4 x i32> %tmp19, i32 0
  ret i32 %tmp21
}

define void @get_image() nounwind {
; CHECK: @get_image
; CHECK-NOT: extractelement
; CHECK: unreachable
entry:
  %0 = call i32 @fgetc(i8* null) nounwind               ; <i32> [#uses=1]
  %1 = trunc i32 %0 to i8         ; <i8> [#uses=1]
  %tmp2 = insertelement <100 x i8> zeroinitializer, i8 %1, i32 1          ; <<100 x i8>> [#uses=1]
  %tmp1 = extractelement <100 x i8> %tmp2, i32 0          ; <i8> [#uses=1]
  %2 = icmp eq i8 %tmp1, 80               ; <i1> [#uses=1]
  br i1 %2, label %bb2, label %bb3

bb2:            ; preds = %entry
  br label %bb3

bb3:            ; preds = %bb2, %entry
  unreachable
}

; PR4340
define void @vac(<4 x float>* nocapture %a) nounwind {
; CHECK: @vac
; CHECK-NOT: load
; CHECK: ret
entry:
	%tmp1 = load <4 x float>* %a		; <<4 x float>> [#uses=1]
	%vecins = insertelement <4 x float> %tmp1, float 0.000000e+00, i32 0	; <<4 x float>> [#uses=1]
	%vecins4 = insertelement <4 x float> %vecins, float 0.000000e+00, i32 1; <<4 x float>> [#uses=1]
	%vecins6 = insertelement <4 x float> %vecins4, float 0.000000e+00, i32 2; <<4 x float>> [#uses=1]
	%vecins8 = insertelement <4 x float> %vecins6, float 0.000000e+00, i32 3; <<4 x float>> [#uses=1]
	store <4 x float> %vecins8, <4 x float>* %a
	ret void
}

declare i32 @fgetc(i8*)

declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>)

declare i32 @llvm.x86.sse.cvttss2si(<4 x float>)
