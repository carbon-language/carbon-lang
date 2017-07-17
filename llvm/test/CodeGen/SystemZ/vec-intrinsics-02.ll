; Test vector intrinsics added with z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare <2 x i64> @llvm.s390.vbperm(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vmslg(<2 x i64>, <2 x i64>, <16 x i8>, i32)
declare <16 x i8> @llvm.s390.vlrl(i32, i8 *)
declare void @llvm.s390.vstrl(<16 x i8>, i32, i8 *)
declare <2 x double> @llvm.s390.vfmaxdb(<2 x double>, <2 x double>, i32)
declare <2 x double> @llvm.s390.vfmindb(<2 x double>, <2 x double>, i32)

; VBPERM.
define <2 x i64> @test_vbperm(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vbperm:
; CHECK: vbperm %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vbperm(<16 x i8> %a, <16 x i8> %b)
  ret <2 x i64> %res
}

; VMSLG with no shifts.
define <16 x i8> @test_vmslg1(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmslg1:
; CHECK: vmslg %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmslg(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c, i32 0)
  ret <16 x i8> %res
}

; VMSLG with both shifts.
define <16 x i8> @test_vmslg2(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmslg2:
; CHECK: vmslg %v24, %v24, %v26, %v28, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmslg(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c, i32 12)
  ret <16 x i8> %res
}

; VLRLR with the lowest in-range displacement.
define <16 x i8> @test_vlrlr1(i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vlrlr1:
; CHECK: vlrlr %v24, %r3, 0(%r2)
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRLR with the highest in-range displacement.
define <16 x i8> @test_vlrlr2(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vlrlr2:
; CHECK: vlrlr %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRLR with an out-of-range displacement.
define <16 x i8> @test_vlrlr3(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vlrlr3:
; CHECK: vlrlr %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; Check that VLRLR doesn't allow an index.
define <16 x i8> @test_vlrlr4(i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vlrlr4:
; CHECK: vlrlr %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with the lowest in-range displacement.
define <16 x i8> @test_vlrl1(i8 *%ptr) {
; CHECK-LABEL: test_vlrl1:
; CHECK: vlrl %v24, 0(%r2), 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with the highest in-range displacement.
define <16 x i8> @test_vlrl2(i8 *%base) {
; CHECK-LABEL: test_vlrl2:
; CHECK: vlrl %v24, 4095(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with an out-of-range displacement.
define <16 x i8> @test_vlrl3(i8 *%base) {
; CHECK-LABEL: test_vlrl3:
; CHECK: vlrl %v24, 0({{%r[1-5]}}), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; Check that VLRL doesn't allow an index.
define <16 x i8> @test_vlrl4(i8 *%base, i64 %index) {
; CHECK-LABEL: test_vlrl4:
; CHECK: vlrl %v24, 0({{%r[1-5]}}), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VSTRLR with the lowest in-range displacement.
define void @test_vstrlr1(<16 x i8> %vec, i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vstrlr1:
; CHECK: vstrlr %v24, %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRLR with the highest in-range displacement.
define void @test_vstrlr2(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstrlr2:
; CHECK: vstrlr %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRLR with an out-of-range displacement.
define void @test_vstrlr3(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstrlr3:
; CHECK: vstrlr %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; Check that VSTRLR doesn't allow an index.
define void @test_vstrlr4(<16 x i8> %vec, i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vstrlr4:
; CHECK: vstrlr %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRL with the lowest in-range displacement.
define void @test_vstrl1(<16 x i8> %vec, i8 *%ptr) {
; CHECK-LABEL: test_vstrl1:
; CHECK: vstrl %v24, 0(%r2), 8
; CHECK: br %r14
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VSTRL with the highest in-range displacement.
define void @test_vstrl2(<16 x i8> %vec, i8 *%base) {
; CHECK-LABEL: test_vstrl2:
; CHECK: vstrl %v24, 4095(%r2), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VSTRL with an out-of-range displacement.
define void @test_vstrl3(<16 x i8> %vec, i8 *%base) {
; CHECK-LABEL: test_vstrl3:
; CHECK: vstrl %v24, 0({{%r[1-5]}}), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; Check that VSTRL doesn't allow an index.
define void @test_vstrl4(<16 x i8> %vec, i8 *%base, i64 %index) {
; CHECK-LABEL: test_vstrl4:
; CHECK: vstrl %v24, 0({{%r[1-5]}}), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VFMAXDB.
define <2 x double> @test_vfmaxdb(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfmaxdb:
; CHECK: vfmaxdb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %a, <2 x double> %b, i32 4)
  ret <2 x double> %res
}

; VFMINDB.
define <2 x double> @test_vfmindb(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfmindb:
; CHECK: vfmindb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfmindb(<2 x double> %a, <2 x double> %b, i32 4)
  ret <2 x double> %res
}

