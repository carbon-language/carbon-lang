; RUN: llc < %s -march=x86-64 -mcpu=penryn | FileCheck -check-prefix=SSE41 %s
; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx | FileCheck -check-prefix=AVX1 %s
; RUN: llc < %s -march=x86-64 -mcpu=core-avx2 | FileCheck -check-prefix=AVX2 %s

; PR14887
; These tests inject a store into the chain to test the inreg versions of pmovsx

define void @test1(<2 x i8>* %in, <2 x i64>* %out) nounwind {
  %wide.load35 = load <2 x i8>* %in, align 1
  %sext = sext <2 x i8> %wide.load35 to <2 x i64>
  store <2 x i64> zeroinitializer, <2 x i64>* undef, align 8
  store <2 x i64> %sext, <2 x i64>* %out, align 8
  ret void

; SSE41: test1:
; SSE41: pmovsxbq

; AVX1: test1:
; AVX1: vpmovsxbq

; AVX2: test1:
; AVX2: vpmovsxbq
}

define void @test2(<4 x i8>* %in, <4 x i64>* %out) nounwind {
  %wide.load35 = load <4 x i8>* %in, align 1
  %sext = sext <4 x i8> %wide.load35 to <4 x i64>
  store <4 x i64> zeroinitializer, <4 x i64>* undef, align 8
  store <4 x i64> %sext, <4 x i64>* %out, align 8
  ret void

; AVX2: test2:
; AVX2: vpmovsxbq
}

define void @test3(<4 x i8>* %in, <4 x i32>* %out) nounwind {
  %wide.load35 = load <4 x i8>* %in, align 1
  %sext = sext <4 x i8> %wide.load35 to <4 x i32>
  store <4 x i32> zeroinitializer, <4 x i32>* undef, align 8
  store <4 x i32> %sext, <4 x i32>* %out, align 8
  ret void

; SSE41: test3:
; SSE41: pmovsxbd

; AVX1: test3:
; AVX1: vpmovsxbd

; AVX2: test3:
; AVX2: vpmovsxbd
}

define void @test4(<8 x i8>* %in, <8 x i32>* %out) nounwind {
  %wide.load35 = load <8 x i8>* %in, align 1
  %sext = sext <8 x i8> %wide.load35 to <8 x i32>
  store <8 x i32> zeroinitializer, <8 x i32>* undef, align 8
  store <8 x i32> %sext, <8 x i32>* %out, align 8
  ret void

; AVX2: test4:
; AVX2: vpmovsxbd
}

define void @test5(<8 x i8>* %in, <8 x i16>* %out) nounwind {
  %wide.load35 = load <8 x i8>* %in, align 1
  %sext = sext <8 x i8> %wide.load35 to <8 x i16>
  store <8 x i16> zeroinitializer, <8 x i16>* undef, align 8
  store <8 x i16> %sext, <8 x i16>* %out, align 8
  ret void

; SSE41: test5:
; SSE41: pmovsxbw

; AVX1: test5:
; AVX1: vpmovsxbw

; AVX2: test5:
; AVX2: vpmovsxbw
}

define void @test6(<16 x i8>* %in, <16 x i16>* %out) nounwind {
  %wide.load35 = load <16 x i8>* %in, align 1
  %sext = sext <16 x i8> %wide.load35 to <16 x i16>
  store <16 x i16> zeroinitializer, <16 x i16>* undef, align 8
  store <16 x i16> %sext, <16 x i16>* %out, align 8
  ret void

; AVX2: test6:
; FIXME: v16i8 -> v16i16 is scalarized.
; AVX2-NOT: pmovsx
}

define void @test7(<2 x i16>* %in, <2 x i64>* %out) nounwind {
  %wide.load35 = load <2 x i16>* %in, align 1
  %sext = sext <2 x i16> %wide.load35 to <2 x i64>
  store <2 x i64> zeroinitializer, <2 x i64>* undef, align 8
  store <2 x i64> %sext, <2 x i64>* %out, align 8
  ret void


; SSE41: test7:
; SSE41: pmovsxwq

; AVX1: test7:
; AVX1: vpmovsxwq

; AVX2: test7:
; AVX2: vpmovsxwq
}

define void @test8(<4 x i16>* %in, <4 x i64>* %out) nounwind {
  %wide.load35 = load <4 x i16>* %in, align 1
  %sext = sext <4 x i16> %wide.load35 to <4 x i64>
  store <4 x i64> zeroinitializer, <4 x i64>* undef, align 8
  store <4 x i64> %sext, <4 x i64>* %out, align 8
  ret void

; AVX2: test8:
; AVX2: vpmovsxwq
}

define void @test9(<4 x i16>* %in, <4 x i32>* %out) nounwind {
  %wide.load35 = load <4 x i16>* %in, align 1
  %sext = sext <4 x i16> %wide.load35 to <4 x i32>
  store <4 x i32> zeroinitializer, <4 x i32>* undef, align 8
  store <4 x i32> %sext, <4 x i32>* %out, align 8
  ret void

; SSE41: test9:
; SSE41: pmovsxwd

; AVX1: test9:
; AVX1: vpmovsxwd

; AVX2: test9:
; AVX2: vpmovsxwd
}

define void @test10(<8 x i16>* %in, <8 x i32>* %out) nounwind {
  %wide.load35 = load <8 x i16>* %in, align 1
  %sext = sext <8 x i16> %wide.load35 to <8 x i32>
  store <8 x i32> zeroinitializer, <8 x i32>* undef, align 8
  store <8 x i32> %sext, <8 x i32>* %out, align 8
  ret void

; AVX2: test10:
; AVX2: vpmovsxwd
}

define void @test11(<2 x i32>* %in, <2 x i64>* %out) nounwind {
  %wide.load35 = load <2 x i32>* %in, align 1
  %sext = sext <2 x i32> %wide.load35 to <2 x i64>
  store <2 x i64> zeroinitializer, <2 x i64>* undef, align 8
  store <2 x i64> %sext, <2 x i64>* %out, align 8
  ret void

; SSE41: test11:
; SSE41: pmovsxdq

; AVX1: test11:
; AVX1: vpmovsxdq

; AVX2: test11:
; AVX2: vpmovsxdq
}

define void @test12(<4 x i32>* %in, <4 x i64>* %out) nounwind {
  %wide.load35 = load <4 x i32>* %in, align 1
  %sext = sext <4 x i32> %wide.load35 to <4 x i64>
  store <4 x i64> zeroinitializer, <4 x i64>* undef, align 8
  store <4 x i64> %sext, <4 x i64>* %out, align 8
  ret void

; AVX2: test12:
; AVX2: vpmovsxdq
}
