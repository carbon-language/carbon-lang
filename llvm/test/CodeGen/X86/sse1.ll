; Tests for SSE1 and below, without SSE2+.
; RUN: llc < %s -march=x86 -mcpu=pentium3 -O3 | FileCheck %s
; RUN: llc < %s -march=x86-64 -mcpu=pentium3 -O3 | FileCheck %s

define <8 x i16> @test1(<8 x i32> %a) nounwind {
; CHECK: test1
  ret <8 x i16> zeroinitializer
}

define <8 x i16> @test2(<8 x i32> %a) nounwind {
; CHECK: test2
  %c = trunc <8 x i32> %a to <8 x i16>            ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %c
}

; PR7993
;define <4 x i32> @test3(<4 x i16> %a) nounwind {
;  %c = sext <4 x i16> %a to <4 x i32>             ; <<4 x i32>> [#uses=1]
;  ret <4 x i32> %c
;}
