; RUN: llc < %s -march=x86-64 -mattr=+sse41 -asm-verbose=0 | FileCheck %s

define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test1:
; CHECK-NEXT: pmulld
  %C = mul <4 x i32> %A, %B
  ret <4 x i32> %C
}

define <4 x i32> @test1a(<4 x i32> %A, <4 x i32> *%Bp) nounwind {
; CHECK: test1a:
; CHECK-NEXT: pmulld
  %B = load <4 x i32>* %Bp
  %C = mul <4 x i32> %A, %B
  ret <4 x i32> %C
}
