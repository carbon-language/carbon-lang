; Tests for SSE1 and below, without SSE2+.
; RUN: llc < %s -mcpu=pentium3 -O3 | FileCheck %s

define <8 x i16> @test1(<8 x i32> %a) nounwind {
; CHECK: test1
  ret <8 x i16> zeroinitializer
}
