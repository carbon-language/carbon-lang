; RUN: llc -march=x86-64 -mcpu=corei7 -mattr=-sse4.1 < %s | FileCheck %s

; Verify that we correctly fold target specific packed vector shifts by
; immediate count into a simple build_vector when the elements of the vector
; in input to the packed shift are all constants or undef.

define <8 x i16> @test1() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> <i16 1, i16 2, i16 4, i16 8, i16 1, i16 2, i16 4, i16 8>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test1
; CHECK-NOT: psll
; CHECK: movaps
; CHECK-NEXT: ret

define <8 x i16> @test2() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> <i16 4, i16 8, i16 16, i16 32, i16 4, i16 8, i16 16, i16 32>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test2
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <8 x i16> @test3() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> <i16 4, i16 8, i16 16, i16 32, i16 4, i16 8, i16 16, i16 32>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test3
; CHECK-NOT: psra
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test4() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> <i32 1, i32 2, i32 4, i32 8>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test4
; CHECK-NOT: psll
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test5() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> <i32 4, i32 8, i32 16, i32 32>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test5
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test6() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> <i32 4, i32 8, i32 16, i32 32>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test6
; CHECK-NOT: psra
; CHECK: movaps
; CHECK-NEXT: ret

define <2 x i64> @test7() {
  %1 = tail call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> <i64 1, i64 2>, i32 3)
  ret <2 x i64> %1
}
; CHECK-LABEL: test7
; CHECK-NOT: psll
; CHECK: movaps
; CHECK-NEXT: ret

define <2 x i64> @test8() {
  %1 = tail call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> <i64 8, i64 16>, i32 3)
  ret <2 x i64> %1
}
; CHECK-LABEL: test8
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <8 x i16> @test9() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> <i16 15, i16 8, i16 undef, i16 undef, i16 31, i16 undef, i16 64, i16 128>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test9
; CHECK-NOT: psra
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test10() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> <i32 undef, i32 8, i32 undef, i32 32>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test10
; CHECK-NOT: psra
; CHECK: movaps
; CHECK-NEXT: ret

define <2 x i64> @test11() {
  %1 = tail call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> <i64 undef, i64 31>, i32 3)
  ret <2 x i64> %1
}
; CHECK-LABEL: test11
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <8 x i16> @test12() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> <i16 15, i16 8, i16 undef, i16 undef, i16 31, i16 undef, i16 64, i16 128>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test12
; CHECK-NOT: psra
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test13() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> <i32 undef, i32 8, i32 undef, i32 32>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test13
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <8 x i16> @test14() {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> <i16 15, i16 8, i16 undef, i16 undef, i16 31, i16 undef, i16 64, i16 128>, i32 3)
  ret <8 x i16> %1
}
; CHECK-LABEL: test14
; CHECK-NOT: psrl
; CHECK: movaps
; CHECK-NEXT: ret

define <4 x i32> @test15() {
  %1 = tail call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> <i32 undef, i32 8, i32 undef, i32 32>, i32 3)
  ret <4 x i32> %1
}
; CHECK-LABEL: test15
; CHECK-NOT: psll
; CHECK: movaps
; CHECK-NEXT: ret

define <2 x i64> @test16() {
  %1 = tail call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> <i64 undef, i64 31>, i32 3)
  ret <2 x i64> %1
}
; CHECK-LABEL: test16
; CHECK-NOT: psll
; CHECK: movaps
; CHECK-NEXT: ret


declare <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16>, i32)
declare <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16>, i32)
declare <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16>, i32)
declare <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32>, i32)
declare <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32>, i32)
declare <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32>, i32)
declare <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64>, i32)
declare <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64>, i32)

