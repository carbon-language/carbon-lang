; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

define <8 x i16> @test_zext_load() {
  ; CHECK: vmovq
entry:
  %0 = load <2 x i16> ** undef, align 8
  %1 = getelementptr inbounds <2 x i16>* %0, i64 1
  %2 = load <2 x i16>* %0, align 1
  %3 = shufflevector <2 x i16> %2, <2 x i16> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = load <2 x i16>* %1, align 1
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <8 x i16> %3, <8 x i16> %5, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %6
}
