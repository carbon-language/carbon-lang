; RUN: opt < %s -instcombine -S | FileCheck %s

%T = type <4 x float>


define %T @test1(%T %v1) {
; CHECK: @test1
; CHECK: ret %T %v1
  %v2 = shufflevector %T %v1, %T undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret %T %v2
}

define %T @test2(%T %v1) {
; CHECK: @test2
; CHECK: ret %T %v1
  %v2 = shufflevector %T %v1, %T %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret %T %v2
}

define float @test3(%T %A, %T %B, float %f) {
; CHECK: @test3
; CHECK: ret float %f
        %C = insertelement %T %A, float %f, i32 0
        %D = shufflevector %T %C, %T %B, <4 x i32> <i32 5, i32 0, i32 2, i32 7>
        %E = extractelement %T %D, i32 1
        ret float %E
}

define i32 @test4(<4 x i32> %X) {
; CHECK: @test4
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define i32 @test5(<4 x i32> %X) {
; CHECK: @test5
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 undef, i32 undef>
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define float @test6(<4 x float> %X) {
; CHECK: @test6
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %X1 = bitcast <4 x float> %X to <4 x i32>
        %tmp152.i53899.i = shufflevector <4 x i32> %X1, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp152.i53900.i = bitcast <4 x i32> %tmp152.i53899.i to <4 x float>
        %tmp34 = extractelement <4 x float> %tmp152.i53900.i, i32 0
        ret float %tmp34
}

define <4 x float> @test7(<4 x float> %tmp45.i) {
; CHECK: @test7
; CHECK-NEXT: ret %T %tmp45.i
        %tmp1642.i = shufflevector <4 x float> %tmp45.i, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 6, i32 7 >
        ret <4 x float> %tmp1642.i
}

; This should turn into a single shuffle.
define <4 x float> @test8(<4 x float> %tmp, <4 x float> %tmp1) {
; CHECK: @test8
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
        %tmp4 = extractelement <4 x float> %tmp, i32 1
        %tmp2 = extractelement <4 x float> %tmp, i32 3
        %tmp1.upgrd.1 = extractelement <4 x float> %tmp1, i32 0
        %tmp128 = insertelement <4 x float> undef, float %tmp4, i32 0
        %tmp130 = insertelement <4 x float> %tmp128, float undef, i32 1
        %tmp132 = insertelement <4 x float> %tmp130, float %tmp2, i32 2 
        %tmp134 = insertelement <4 x float> %tmp132, float %tmp1.upgrd.1, i32 3
        ret <4 x float> %tmp134
}

; Test fold of two shuffles where the first shuffle vectors inputs are a
; different length then the second.
define <4 x i8> @test9(<16 x i8> %tmp6) nounwind {
; CHECK: @test9
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
	%tmp7 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> < i32 13, i32 9, i32 4, i32 13 >		; <<4 x i8>> [#uses=1]
	%tmp9 = shufflevector <4 x i8> %tmp7, <4 x i8> undef, <4 x i32> < i32 3, i32 1, i32 2, i32 0 >		; <<4 x i8>> [#uses=1]
	ret <4 x i8> %tmp9
}

; Test fold of hi/lo vector halves
; Test fold of unpack operation
define void @test10(<16 x i8>* %out, <16 x i8> %r, <16 x i8> %g, <16 x i8> %b, <16 x i8> %a) nounwind ssp {
; CHECK: @test10
; CHECK-NEXT: shufflevector
; CHECK-NEXT: shufflevector
; CHECK-NEXT: store
; CHECK-NEXT: getelementptr
; CHECK-NEXT: store
; CHECK-NEXT: ret
  %tmp1 = shufflevector <16 x i8> %r, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> ; <<8 x i8>> [#uses=1]
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef> ; <<16 x i8>> [#uses=1]
  %tmp4 = shufflevector <16 x i8> undef, <16 x i8> %tmp3, <16 x i32> <i32 16, i32 1, i32 17, i32 3, i32 18, i32 5, i32 19, i32 7, i32 20, i32 9, i32 21, i32 11, i32 22, i32 13, i32 23, i32 15> ; <<16 x i8>> [#uses=1]
  %tmp6 = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> ; <<8 x i8>> [#uses=1]
  %tmp8 = shufflevector <8 x i8> %tmp6, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef> ; <<16 x i8>> [#uses=1]
  %tmp9 = shufflevector <16 x i8> %tmp4, <16 x i8> %tmp8, <16 x i32> <i32 0, i32 16, i32 2, i32 17, i32 4, i32 18, i32 6, i32 19, i32 8, i32 20, i32 10, i32 21, i32 12, i32 22, i32 14, i32 23> ; <<16 x i8>> [#uses=1]
  %tmp11 = shufflevector <16 x i8> %r, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> ; <<8 x i8>> [#uses=1]
  %tmp13 = shufflevector <8 x i8> %tmp11, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef> ; <<16 x i8>> [#uses=1]
  %tmp14 = shufflevector <16 x i8> undef, <16 x i8> %tmp13, <16 x i32> <i32 16, i32 1, i32 17, i32 3, i32 18, i32 5, i32 19, i32 7, i32 20, i32 9, i32 21, i32 11, i32 22, i32 13, i32 23, i32 15> ; <<16 x i8>> [#uses=1]
  %tmp16 = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> ; <<8 x i8>> [#uses=1]
  %tmp18 = shufflevector <8 x i8> %tmp16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef> ; <<16 x i8>> [#uses=1]
  %tmp19 = shufflevector <16 x i8> %tmp14, <16 x i8> %tmp18, <16 x i32> <i32 0, i32 16, i32 2, i32 17, i32 4, i32 18, i32 6, i32 19, i32 8, i32 20, i32 10, i32 21, i32 12, i32 22, i32 14, i32 23> ; <<16 x i8>> [#uses=1]
  %arrayidx = getelementptr inbounds <16 x i8>* %out, i64 0 ; <<16 x i8>*> [#uses=1]
  store <16 x i8> %tmp9, <16 x i8>* %arrayidx
  %arrayidx24 = getelementptr inbounds <16 x i8>* %out, i64 1 ; <<16 x i8>*> [#uses=1]
  store <16 x i8> %tmp19, <16 x i8>* %arrayidx24
  ret void
}
