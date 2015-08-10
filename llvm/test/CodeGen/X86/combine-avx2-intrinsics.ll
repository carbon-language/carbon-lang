; RUN: llc < %s -march=x86-64 -mcpu=core-avx2 | FileCheck %s

; Verify that the backend correctly combines AVX2 builtin intrinsics.


define <32 x i8> @test_x86_avx2_pblendvb(<32 x i8> %a0, <32 x i8> %a1) {
  %res = call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %a0, <32 x i8> %a0, <32 x i8> %a1)
  ret <32 x i8> %res
}
; CHECK-LABEL: test_x86_avx2_pblendvb
; CHECK-NOT: vpblendvb
; CHECK: ret


define <16 x i16> @test_x86_avx2_pblendw(<16 x i16> %a0) {
  %res = call <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16> %a0, <16 x i16> %a0, i32 7)
  ret <16 x i16> %res
}
; CHECK-LABEL: test_x86_avx2_pblendw
; CHECK-NOT: vpblendw
; CHECK: ret


define <4 x i32> @test_x86_avx2_pblendd_128(<4 x i32> %a0) {
  %res = call <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32> %a0, <4 x i32> %a0, i32 7)
  ret <4 x i32> %res
}
; CHECK-LABEL: test_x86_avx2_pblendd_128
; CHECK-NOT: vpblendd
; CHECK: ret


define <8 x i32> @test_x86_avx2_pblendd_256(<8 x i32> %a0) {
  %res = call <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32> %a0, <8 x i32> %a0, i32 7)
  ret <8 x i32> %res
}
; CHECK-LABEL: test_x86_avx2_pblendd_256
; CHECK-NOT: vpblendd
; CHECK: ret


define <32 x i8> @test2_x86_avx2_pblendvb(<32 x i8> %a0, <32 x i8> %a1) {
  %res = call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %a0, <32 x i8> %a1, <32 x i8> zeroinitializer)
  ret <32 x i8> %res
}
; CHECK-LABEL: test2_x86_avx2_pblendvb
; CHECK-NOT: vpblendvb
; CHECK: ret


define <16 x i16> @test2_x86_avx2_pblendw(<16 x i16> %a0, <16 x i16> %a1) {
  %res = call <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16> %a0, <16 x i16> %a1, i32 0)
  ret <16 x i16> %res
}
; CHECK-LABEL: test2_x86_avx2_pblendw
; CHECK-NOT: vpblendw
; CHECK: ret


define <4 x i32> @test2_x86_avx2_pblendd_128(<4 x i32> %a0, <4 x i32> %a1) {
  %res = call <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32> %a0, <4 x i32> %a1, i32 0)
  ret <4 x i32> %res
}
; CHECK-LABEL: test2_x86_avx2_pblendd_128
; CHECK-NOT: vpblendd
; CHECK: ret


define <8 x i32> @test2_x86_avx2_pblendd_256(<8 x i32> %a0, <8 x i32> %a1) {
  %res = call <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32> %a0, <8 x i32> %a1, i32 0)
  ret <8 x i32> %res
}
; CHECK-LABEL: test2_x86_avx2_pblendd_256
; CHECK-NOT: vpblendd
; CHECK: ret


define <32 x i8> @test3_x86_avx2_pblendvb(<32 x i8> %a0, <32 x i8> %a1) {
  %1 = bitcast <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1> to <32 x i8>
  %res = call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %a0, <32 x i8> %a1, <32 x i8> %1)
  ret <32 x i8> %res
}
; CHECK-LABEL: test3_x86_avx2_pblendvb
; CHECK-NOT: vpblendvb
; CHECK: ret


define <16 x i16> @test3_x86_avx2_pblendw(<16 x i16> %a0, <16 x i16> %a1) {
  %res = call <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16> %a0, <16 x i16> %a1, i32 -1)
  ret <16 x i16> %res
}
; CHECK-LABEL: test3_x86_avx2_pblendw
; CHECK-NOT: vpblendw
; CHECK: ret


define <4 x i32> @test3_x86_avx2_pblendd_128(<4 x i32> %a0, <4 x i32> %a1) {
  %res = call <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32> %a0, <4 x i32> %a1, i32 -1)
  ret <4 x i32> %res
}
; CHECK-LABEL: test3_x86_avx2_pblendd_128
; CHECK-NOT: vpblendd
; CHECK: ret


define <8 x i32> @test3_x86_avx2_pblendd_256(<8 x i32> %a0, <8 x i32> %a1) {
  %res = call <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32> %a0, <8 x i32> %a1, i32 -1)
  ret <8 x i32> %res
}
; CHECK-LABEL: test3_x86_avx2_pblendd_256
; CHECK-NOT: vpblendd
; CHECK: ret


declare <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8>, <32 x i8>, <32 x i8>)
declare <16 x i16> @llvm.x86.avx2.pblendw(<16 x i16>, <16 x i16>, i32)
declare <4 x i32> @llvm.x86.avx2.pblendd.128(<4 x i32>, <4 x i32>, i32)
declare <8 x i32> @llvm.x86.avx2.pblendd.256(<8 x i32>, <8 x i32>, i32)

