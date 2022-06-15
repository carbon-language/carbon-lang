; RUN: llc -mtriple aarch64 -mattr=+sve -asm-verbose=1 < %s | FileCheck %s
; RUN: llc -mtriple aarch64 -mattr=+sme -asm-verbose=1 < %s | FileCheck %s

;
; SVCREATE2 (i8)
;

define <vscale x 16 x i8> @test_svcreate2_s8_vec0(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s8_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %tuple, i32 0)
  ret <vscale x 16 x i8> %extract
}

define <vscale x 16 x i8> @test_svcreate2_s8_vec1(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s8_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %tuple, i32 1)
  ret <vscale x 16 x i8> %extract
}

;
; SVCREATE2 (i16)
;

define <vscale x 8 x i16> @test_svcreate2_s16_vec0(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %tuple, i32 0)
  ret <vscale x 8 x i16> %extract
}

define <vscale x 8 x i16> @test_svcreate2_s16_vec1(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s16_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %tuple, i32 1)
  ret <vscale x 8 x i16> %extract
}

;
; SVCREATE2 (half)
;

define <vscale x 8 x half> @test_svcreate2_f16_vec0(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x half> @llvm.aarch64.sve.tuple.create2.nxv16f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half> %tuple, i32 0)
  ret <vscale x 8 x half> %extract
}

define <vscale x 8 x half> @test_svcreate2_f16_vec1(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f16_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x half> @llvm.aarch64.sve.tuple.create2.nxv16f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half> %tuple, i32 1)
  ret <vscale x 8 x half> %extract
}

;
; SVCREATE2 (bfloat)
;

define <vscale x 8 x bfloat> @test_svcreate2_bf16_vec0(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1) #1 {
; CHECK-LABEL: test_svcreate2_bf16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x bfloat> @llvm.aarch64.sve.tuple.create2.nxv16bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> %tuple, i32 0)
  ret <vscale x 8 x bfloat> %extract
}

define <vscale x 8 x bfloat> @test_svcreate2_bf16_vec1(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1) #1 {
; CHECK-LABEL: test_svcreate2_bf16_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x bfloat> @llvm.aarch64.sve.tuple.create2.nxv16bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> %tuple, i32 1)
  ret <vscale x 8 x bfloat> %extract
}

;
; SVCREATE2 (i32)
;

define <vscale x 4 x i32> @test_svcreate2_s32_vec0(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %tuple, i32 0)
  ret <vscale x 4 x i32> %extract
}

define <vscale x 4 x i32> @test_svcreate2_s32_vec1(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s32_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %tuple, i32 1)
  ret <vscale x 4 x i32> %extract
}

;
; SVCREATE2 (float)
;

define <vscale x 4 x float> @test_svcreate2_f32_vec0(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float> %tuple, i32 0)
  ret <vscale x 4 x float> %extract
}

define <vscale x 4 x float> @test_svcreate2_f32_vec1(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f32_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float> %tuple, i32 1)
  ret <vscale x 4 x float> %extract
}

;
; SVCREATE2 (i64)
;

define <vscale x 2 x i64> @test_svcreate2_s64_vec0(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %tuple, i32 0)
  ret <vscale x 2 x i64> %extract
}

define <vscale x 2 x i64> @test_svcreate2_s64_vec1(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1) #0 {
; CHECK-LABEL: test_svcreate2_s64_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %tuple, i32 1)
  ret <vscale x 2 x i64> %extract
}

;
; SVCREATE2 (double)
;

define <vscale x 2 x double> @test_svcreate2_f64_vec0(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double> %tuple, i32 0)
  ret <vscale x 2 x double> %extract
}

define <vscale x 2 x double> @test_svcreate2_f64_vec1(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1) #0 {
; CHECK-LABEL: test_svcreate2_f64_vec1:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double> %tuple, i32 1)
  ret <vscale x 2 x double> %extract
}

;
; SVCREATE3 (i8)
;

define <vscale x 16 x i8> @test_svcreate3_s8_vec0(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s8_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv48i8(<vscale x 48 x i8> %tuple, i32 0)
  ret <vscale x 16 x i8> %extract
}

define <vscale x 16 x i8> @test_svcreate3_s8_vec2(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s8_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv48i8(<vscale x 48 x i8> %tuple, i32 2)
  ret <vscale x 16 x i8> %extract
}

;
; SVCREATE3 (i16)
;

define <vscale x 8 x i16> @test_svcreate3_s16_vec0(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv24i16(<vscale x 24 x i16> %tuple, i32 0)
  ret <vscale x 8 x i16> %extract
}

define <vscale x 8 x i16> @test_svcreate3_s16_vec2(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s16_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv24i16(<vscale x 24 x i16> %tuple, i32 2)
  ret <vscale x 8 x i16> %extract
}

;
; SVCREATE3 (half)
;

define <vscale x 8 x half> @test_svcreate3_f16_vec0(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x half> @llvm.aarch64.sve.tuple.create3.nxv24f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv24f16(<vscale x 24 x half> %tuple, i32 0)
  ret <vscale x 8 x half> %extract
}

define <vscale x 8 x half> @test_svcreate3_f16_vec2(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f16_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x half> @llvm.aarch64.sve.tuple.create3.nxv24f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv24f16(<vscale x 24 x half> %tuple, i32 2)
  ret <vscale x 8 x half> %extract
}

;
; SVCREATE3 (bfloat)
;

define <vscale x 8 x bfloat> @test_svcreate3_bf16_vec0(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2) #1 {
; CHECK-LABEL: test_svcreate3_bf16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x bfloat> @llvm.aarch64.sve.tuple.create3.nxv24bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %tuple, i32 0)
  ret <vscale x 8 x bfloat> %extract
}

define <vscale x 8 x bfloat> @test_svcreate3_bf16_vec2(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2) #1 {
; CHECK-LABEL: test_svcreate3_bf16_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x bfloat> @llvm.aarch64.sve.tuple.create3.nxv24bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %tuple, i32 2)
  ret <vscale x 8 x bfloat> %extract
}

;
; SVCREATE3 (i32)
;

define <vscale x 4 x i32> @test_svcreate3_s32_vec0(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv12i32(<vscale x 12 x i32> %tuple, i32 0)
  ret <vscale x 4 x i32> %extract
}

define <vscale x 4 x i32> @test_svcreate3_s32_vec2(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s32_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv12i32(<vscale x 12 x i32> %tuple, i32 2)
  ret <vscale x 4 x i32> %extract
}

;
; SVCREATE3 (float)
;

define <vscale x 4 x float> @test_svcreate3_f32_vec0(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv12f32(<vscale x 12 x float> %tuple, i32 0)
  ret <vscale x 4 x float> %extract
}

define <vscale x 4 x float> @test_svcreate3_f32_vec2(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f32_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv12f32(<vscale x 12 x float> %tuple, i32 2)
  ret <vscale x 4 x float> %extract
}

;
; SVCREATE3 (i64)
;

define <vscale x 2 x i64> @test_svcreate3_s64_vec0(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv6i64(<vscale x 6 x i64> %tuple, i32 0)
  ret <vscale x 2 x i64> %extract
}

define <vscale x 2 x i64> @test_svcreate3_s64_vec2(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2) #0 {
; CHECK-LABEL: test_svcreate3_s64_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv6i64(<vscale x 6 x i64> %tuple, i32 2)
  ret <vscale x 2 x i64> %extract
}

;
; SVCREATE3 (double)
;

define <vscale x 2 x double> @test_svcreate3_f64_vec0(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv6f64(<vscale x 6 x double> %tuple, i32 0)
  ret <vscale x 2 x double> %extract
}

define <vscale x 2 x double> @test_svcreate3_f64_vec2(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2) #0 {
; CHECK-LABEL: test_svcreate3_f64_vec2:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv6f64(<vscale x 6 x double> %tuple, i32 2)
  ret <vscale x 2 x double> %extract
}

;
; SVCREATE4 (i8)
;

define <vscale x 16 x i8> @test_svcreate4_s8_vec0(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s8_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv64i8(<vscale x 64 x i8> %tuple, i32 0)
  ret <vscale x 16 x i8> %extract
}

define <vscale x 16 x i8> @test_svcreate4_s8_vec3(i1 %p, <vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s8_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8> %z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 16 x i8> undef
L2:
  %extract = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv64i8(<vscale x 64 x i8> %tuple, i32 3)
  ret <vscale x 16 x i8> %extract
}

;
; SVCREATE4 (i16)
;

define <vscale x 8 x i16> @test_svcreate4_s16_vec0(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv32i16(<vscale x 32 x i16> %tuple, i32 0)
  ret <vscale x 8 x i16> %extract
}

define <vscale x 8 x i16> @test_svcreate4_s16_vec3(i1 %p, <vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s16_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16> %z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x i16> undef
L2:
  %extract = tail call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv32i16(<vscale x 32 x i16> %tuple, i32 3)
  ret <vscale x 8 x i16> %extract
}

;
; SVCREATE4 (half)
;

define <vscale x 8 x half> @test_svcreate4_f16_vec0(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2, <vscale x 8 x half> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x half> @llvm.aarch64.sve.tuple.create4.nxv32f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2, <vscale x 8 x half> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv32f16(<vscale x 32 x half> %tuple, i32 0)
  ret <vscale x 8 x half> %extract
}

define <vscale x 8 x half> @test_svcreate4_f16_vec3(i1 %p, <vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2, <vscale x 8 x half> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f16_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x half> @llvm.aarch64.sve.tuple.create4.nxv32f16.nxv8f16(<vscale x 8 x half> %z0, <vscale x 8 x half> %z1, <vscale x 8 x half> %z2, <vscale x 8 x half> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x half> undef
L2:
  %extract = tail call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv32f16(<vscale x 32 x half> %tuple, i32 3)
  ret <vscale x 8 x half> %extract
}

;
; SVCREATE4 (bfloat)
;

define <vscale x 8 x bfloat> @test_svcreate4_bf16_vec0(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2, <vscale x 8 x bfloat> %z3) #1 {
; CHECK-LABEL: test_svcreate4_bf16_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x bfloat> @llvm.aarch64.sve.tuple.create4.nxv32bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2, <vscale x 8 x bfloat> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> %tuple, i32 0)
  ret <vscale x 8 x bfloat> %extract
}

define <vscale x 8 x bfloat> @test_svcreate4_bf16_vec3(i1 %p, <vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2, <vscale x 8 x bfloat> %z3) #1 {
; CHECK-LABEL: test_svcreate4_bf16_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x bfloat> @llvm.aarch64.sve.tuple.create4.nxv32bf16.nxv8bf16(<vscale x 8 x bfloat> %z0, <vscale x 8 x bfloat> %z1, <vscale x 8 x bfloat> %z2, <vscale x 8 x bfloat> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 8 x bfloat> undef
L2:
  %extract = tail call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> %tuple, i32 3)
  ret <vscale x 8 x bfloat> %extract
}

;
; SVCREATE4 (i32)
;

define <vscale x 4 x i32> @test_svcreate4_s32_vec0(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv16i32(<vscale x 16 x i32> %tuple, i32 0)
  ret <vscale x 4 x i32> %extract
}

define <vscale x 4 x i32> @test_svcreate4_s32_vec3(i1 %p, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s32_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x i32> undef
L2:
  %extract = tail call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv16i32(<vscale x 16 x i32> %tuple, i32 3)
  ret <vscale x 4 x i32> %extract
}

;
; SVCREATE4 (float)
;

define <vscale x 4 x float> @test_svcreate4_f32_vec0(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f32_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x float> @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv16f32(<vscale x 16 x float> %tuple, i32 0)
  ret <vscale x 4 x float> %extract
}

define <vscale x 4 x float> @test_svcreate4_f32_vec3(i1 %p, <vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f32_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x float> @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float> %z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 4 x float> undef
L2:
  %extract = tail call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv16f32(<vscale x 16 x float> %tuple, i32 3)
  ret <vscale x 4 x float> %extract
}

;
; SVCREATE4 (i64)
;

define <vscale x 2 x i64> @test_svcreate4_s64_vec0(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv8i64(<vscale x 8 x i64> %tuple, i32 0)
  ret <vscale x 2 x i64> %extract
}

define <vscale x 2 x i64> @test_svcreate4_s64_vec3(i1 %p, <vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3) #0 {
; CHECK-LABEL: test_svcreate4_s64_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64> %z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x i64> undef
L2:
  %extract = tail call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv8i64(<vscale x 8 x i64> %tuple, i32 3)
  ret <vscale x 2 x i64> %extract
}

;
; SVCREATE4 (double)
;

define <vscale x 2 x double> @test_svcreate4_f64_vec0(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f64_vec0:
; CHECK: // %L2
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x double> @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv8f64(<vscale x 8 x double> %tuple, i32 0)
  ret <vscale x 2 x double> %extract
}

define <vscale x 2 x double> @test_svcreate4_f64_vec3(i1 %p, <vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3) #0 {
; CHECK-LABEL: test_svcreate4_f64_vec3:
; CHECK: // %L2
; CHECK-NEXT: mov z0.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x double> @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64(<vscale x 2 x double> %z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3)
  br i1 %p, label %L1, label %L2
L1:
  ret <vscale x 2 x double> undef
L2:
  %extract = tail call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv8f64(<vscale x 8 x double> %tuple, i32 3)
  ret <vscale x 2 x double> %extract
}

attributes #0 = { nounwind }
; +bf16 is required for the bfloat version.
attributes #1 = { nounwind "target-features"="+bf16" }

declare <vscale x 4 x double>  @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 8 x float>  @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 16 x half>  @llvm.aarch64.sve.tuple.create2.nxv16f16.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 16 x bfloat>  @llvm.aarch64.sve.tuple.create2.nxv16bf16.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 4 x i64>  @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x i32>  @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 32 x i8>  @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 6 x double>  @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 24 x half> @llvm.aarch64.sve.tuple.create3.nxv24f16.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 24 x bfloat> @llvm.aarch64.sve.tuple.create3.nxv24bf16.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 6 x i64>  @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 48 x i8>  @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 8 x double>  @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64 (<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 16 x float>  @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 32 x half>  @llvm.aarch64.sve.tuple.create4.nxv32f16.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 32 x bfloat>  @llvm.aarch64.sve.tuple.create4.nxv32bf16.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x  8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 64 x i8>  @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8>, i32 immarg)
declare <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv48i8(<vscale x 48 x i8>, i32 immarg)
declare <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv64i8(<vscale x 64 x i8>, i32 immarg)

declare <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16>, i32 immarg)
declare <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv24i16(<vscale x 24 x i16>, i32 immarg)
declare <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv32i16(<vscale x 32 x i16>, i32 immarg)

declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32>, i32 immarg)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv12i32(<vscale x 12 x i32>, i32 immarg)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv16i32(<vscale x 16 x i32>, i32 immarg)

declare <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64>, i32 immarg)
declare <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv6i64(<vscale x 6 x i64>, i32 immarg)
declare <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv8i64(<vscale x 8 x i64>, i32 immarg)

declare <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat>, i32 immarg)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat>, i32 immarg)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat>, i32 immarg)

declare <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half>, i32 immarg)
declare <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv24f16(<vscale x 24 x half>, i32 immarg)
declare <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv32f16(<vscale x 32 x half>, i32 immarg)

declare <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float>, i32 immarg)
declare <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv12f32(<vscale x 12 x float>, i32 immarg)
declare <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv16f32(<vscale x 16 x float>, i32 immarg)

declare <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double>, i32 immarg)
declare <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv6f64(<vscale x 6 x double>, i32 immarg)
declare <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv8f64(<vscale x 8 x double>, i32 immarg)
