; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

;Check for a post-increment updating load.
define <4 x i16> @test_vld1_fx_update(i16** %ptr) nounwind {
; CHECK: test_vld1_fx_update
; CHECK: ld1 { v{{[0-9]+}}.4h }, [x{{[0-9]+|sp}}], #8
  %A = load i16** %ptr
  %tmp0 = bitcast i16* %A to i8*
  %tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %tmp0, i32 2)
  %tmp2 = getelementptr i16* %A, i32 4
  store i16* %tmp2, i16** %ptr
  ret <4 x i16> %tmp1
}

;Check for a post-increment updating load with register increment.
define <2 x i32> @test_vld1_reg_update(i32** %ptr, i32 %inc) nounwind {
; CHECK: test_vld1_reg_update
; CHECK: ld1 { v{{[0-9]+}}.2s }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i32** %ptr
  %tmp0 = bitcast i32* %A to i8*
  %tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %tmp0, i32 4)
  %tmp2 = getelementptr i32* %A, i32 %inc
  store i32* %tmp2, i32** %ptr
  ret <2 x i32> %tmp1
}

define <2 x float> @test_vld2_fx_update(float** %ptr) nounwind {
; CHECK: test_vld2_fx_update
; CHECK: ld2 { v{{[0-9]+}}.2s, v{{[0-9]+}}.2s }, [x{{[0-9]+|sp}}], #16
  %A = load float** %ptr
  %tmp0 = bitcast float* %A to i8*
  %tmp1 = call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8* %tmp0, i32 4)
  %tmp2 = extractvalue { <2 x float>, <2 x float> } %tmp1, 0
  %tmp3 = getelementptr float* %A, i32 4
  store float* %tmp3, float** %ptr
  ret <2 x float> %tmp2
}

define <16 x i8> @test_vld2_reg_update(i8** %ptr, i32 %inc) nounwind {
; CHECK: test_vld2_reg_update
; CHECK: ld2 { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i8** %ptr
  %tmp0 = call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8* %A, i32 1)
  %tmp1 = extractvalue { <16 x i8>, <16 x i8> } %tmp0, 0
  %tmp2 = getelementptr i8* %A, i32 %inc
  store i8* %tmp2, i8** %ptr
  ret <16 x i8> %tmp1
}

define <4 x i32> @test_vld3_fx_update(i32** %ptr) nounwind {
; CHECK: test_vld3_fx_update
; CHECK: ld3 { v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s }, [x{{[0-9]+|sp}}], #48
  %A = load i32** %ptr
  %tmp0 = bitcast i32* %A to i8*
  %tmp1 = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8* %tmp0, i32 4)
  %tmp2 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %tmp1, 0
  %tmp3 = getelementptr i32* %A, i32 12
  store i32* %tmp3, i32** %ptr
  ret <4 x i32> %tmp2
}

define <4 x i16> @test_vld3_reg_update(i16** %ptr, i32 %inc) nounwind {
; CHECK: test_vld3_reg_update
; CHECK: ld3 { v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i16** %ptr
  %tmp0 = bitcast i16* %A to i8*
  %tmp1 = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8* %tmp0, i32 2)
  %tmp2 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %tmp1, 0
  %tmp3 = getelementptr i16* %A, i32 %inc
  store i16* %tmp3, i16** %ptr
  ret <4 x i16> %tmp2
}

define <8 x i16> @test_vld4_fx_update(i16** %ptr) nounwind {
; CHECK: test_vld4_fx_update
; CHECK: ld4 { v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h }, [x{{[0-9]+|sp}}], #64
  %A = load i16** %ptr
  %tmp0 = bitcast i16* %A to i8*
  %tmp1 = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8* %tmp0, i32 8)
  %tmp2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %tmp1, 0
  %tmp3 = getelementptr i16* %A, i32 32
  store i16* %tmp3, i16** %ptr
  ret <8 x i16> %tmp2
}

define <8 x i8> @test_vld4_reg_update(i8** %ptr, i32 %inc) nounwind {
; CHECK: test_vld4_reg_update
; CHECK: ld4 { v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i8** %ptr
  %tmp0 = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8* %A, i32 1)
  %tmp1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %tmp0, 0
  %tmp2 = getelementptr i8* %A, i32 %inc
  store i8* %tmp2, i8** %ptr
  ret <8 x i8> %tmp1
}

define void @test_vst1_fx_update(float** %ptr, <2 x float> %B) nounwind {
; CHECK: test_vst1_fx_update
; CHECK: st1 { v{{[0-9]+}}.2s }, [{{x[0-9]+|sp}}], #8
  %A = load float** %ptr
  %tmp0 = bitcast float* %A to i8*
  call void @llvm.arm.neon.vst1.v2f32(i8* %tmp0, <2 x float> %B, i32 4)
  %tmp2 = getelementptr float* %A, i32 2
  store float* %tmp2, float** %ptr
  ret void
}

define void @test_vst1_reg_update(i16** %ptr, <8 x i16> %B, i32 %inc) nounwind {
; CHECK: test_vst1_reg_update
; CHECK: st1 { v{{[0-9]+}}.8h }, [{{x[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i16** %ptr
  %tmp0 = bitcast i16* %A to i8*
  call void @llvm.arm.neon.vst1.v8i16(i8* %tmp0, <8 x i16> %B, i32 2)
  %tmp1 = getelementptr i16* %A, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret void
}

define void @test_vst2_fx_update(i64** %ptr, <1 x i64> %B) nounwind {
; CHECK: test_vst2_fx_update
; CHECK: st1 { v{{[0-9]+}}.1d, v{{[0-9]+}}.1d }, [{{x[0-9]+|sp}}], #16
  %A = load i64** %ptr
  %tmp0 = bitcast i64* %A to i8*
  call void @llvm.arm.neon.vst2.v1i64(i8* %tmp0, <1 x i64> %B, <1 x i64> %B, i32 8)
  %tmp1 = getelementptr i64* %A, i32 2
  store i64* %tmp1, i64** %ptr
  ret void
}

define void @test_vst2_reg_update(i8** %ptr, <8 x i8> %B, i32 %inc) nounwind {
; CHECK: test_vst2_reg_update
; CHECK: st2 { v{{[0-9]+}}.8b, v{{[0-9]+}}.8b }, [{{x[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i8** %ptr
  call void @llvm.arm.neon.vst2.v8i8(i8* %A, <8 x i8> %B, <8 x i8> %B, i32 4)
  %tmp0 = getelementptr i8* %A, i32 %inc
  store i8* %tmp0, i8** %ptr
  ret void
}

define void @test_vst3_fx_update(i32** %ptr, <2 x i32> %B) nounwind {
; CHECK: test_vst3_fx_update
; CHECK: st3 { v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s }, [{{x[0-9]+|sp}}], #24
  %A = load i32** %ptr
  %tmp0 = bitcast i32* %A to i8*
  call void @llvm.arm.neon.vst3.v2i32(i8* %tmp0, <2 x i32> %B, <2 x i32> %B, <2 x i32> %B, i32 4)
  %tmp1 = getelementptr i32* %A, i32 6
  store i32* %tmp1, i32** %ptr
  ret void
}

define void @test_vst3_reg_update(i16** %ptr, <8 x i16> %B, i32 %inc) nounwind {
; CHECK: test_vst3_reg_update
; CHECK: st3 { v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h }, [{{x[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i16** %ptr
  %tmp0 = bitcast i16* %A to i8*
  call void @llvm.arm.neon.vst3.v8i16(i8* %tmp0, <8 x i16> %B, <8 x i16> %B, <8 x i16> %B, i32 2)
  %tmp1 = getelementptr i16* %A, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret void
}

define void @test_vst4_fx_update(float** %ptr, <4 x float> %B) nounwind {
; CHECK: test_vst4_fx_update
; CHECK: st4 { v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s }, [{{x[0-9]+|sp}}], #64
  %A = load float** %ptr
  %tmp0 = bitcast float* %A to i8*
  call void @llvm.arm.neon.vst4.v4f32(i8* %tmp0, <4 x float> %B, <4 x float> %B, <4 x float> %B, <4 x float> %B, i32 4)
  %tmp1 = getelementptr float* %A, i32 16
  store float* %tmp1, float** %ptr
  ret void
}

define void @test_vst4_reg_update(i8** %ptr, <8 x i8> %B, i32 %inc) nounwind {
; CHECK: test_vst4_reg_update
; CHECK: st4 { v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b }, [{{x[0-9]+|sp}}], x{{[0-9]+}}
  %A = load i8** %ptr
  call void @llvm.arm.neon.vst4.v8i8(i8* %A, <8 x i8> %B, <8 x i8> %B, <8 x i8> %B, <8 x i8> %B, i32 1)
  %tmp0 = getelementptr i8* %A, i32 %inc
  store i8* %tmp0, i8** %ptr
  ret void
}


declare <4 x i16> @llvm.arm.neon.vld1.v4i16(i8*, i32)
declare <2 x i32> @llvm.arm.neon.vld1.v2i32(i8*, i32)
declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8*, i32)
declare { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8*, i32)

declare void @llvm.arm.neon.vst1.v2f32(i8*, <2 x float>, i32)
declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>, i32)
declare void @llvm.arm.neon.vst2.v1i64(i8*, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst2.v8i8(i8*, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst3.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32)

define <16 x i8> @test_vld1x2_fx_update(i8* %a, i8** %ptr) {
; CHECK: test_vld1x2_fx_update
; CHECK: ld1 { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+|sp}}], #32
  %1 = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x2.v16i8(i8* %a, i32 1)
  %2 = extractvalue { <16 x i8>, <16 x i8> } %1, 0
  %tmp1 = getelementptr i8* %a, i32 32
  store i8* %tmp1, i8** %ptr
  ret <16 x i8> %2
}

define <8 x i16> @test_vld1x2_reg_update(i16* %a, i16** %ptr, i32 %inc) {
; CHECK: test_vld1x2_reg_update
; CHECK: ld1 { v{{[0-9]+}}.8h, v{{[0-9]+}}.8h }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x2.v8i16(i8* %1, i32 2)
  %3 = extractvalue { <8 x i16>, <8 x i16> } %2, 0
  %tmp1 = getelementptr i16* %a, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret <8 x i16> %3
}

define <2 x i64> @test_vld1x3_fx_update(i64* %a, i64** %ptr) {
; CHECK: test_vld1x3_fx_update
; CHECK: ld1 { v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d }, [x{{[0-9]+|sp}}], #48
  %1 = bitcast i64* %a to i8*
  %2 = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x3.v2i64(i8* %1, i32 8)
  %3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %2, 0
  %tmp1 = getelementptr i64* %a, i32 6
  store i64* %tmp1, i64** %ptr
  ret  <2 x i64> %3
}

define <8 x i16> @test_vld1x3_reg_update(i16* %a, i16** %ptr, i32 %inc) {
; CHECK: test_vld1x3_reg_update
; CHECK: ld1 { v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x3.v8i16(i8* %1, i32 2)
  %3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %2, 0
  %tmp1 = getelementptr i16* %a, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret <8 x i16> %3
}

define <4 x float> @test_vld1x4_fx_update(float* %a, float** %ptr) {
; CHECK: test_vld1x4_fx_update
; CHECK: ld1 { v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s }, [x{{[0-9]+|sp}}], #64
  %1 = bitcast float* %a to i8*
  %2 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x4.v4f32(i8* %1, i32 4)
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 0
  %tmp1 = getelementptr float* %a, i32 16
  store float* %tmp1, float** %ptr
  ret <4 x float> %3
}

define <8 x i8> @test_vld1x4_reg_update(i8* readonly %a, i8** %ptr, i32 %inc) #0 {
; CHECK: test_vld1x4_reg_update
; CHECK: ld1 { v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x4.v8i8(i8* %a, i32 1)
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %1, 0
  %tmp1 = getelementptr i8* %a, i32 %inc
  store i8* %tmp1, i8** %ptr
  ret <8 x i8> %2
}

define void @test_vst1x2_fx_update(i8* %a, [2 x <16 x i8>] %b.coerce, i8** %ptr) #2 {
; CHECK: test_vst1x2_fx_update
; CHECK: st1 { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+|sp}}], #32
  %1 = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %2 = extractvalue [2 x <16 x i8>] %b.coerce, 1
  tail call void @llvm.aarch64.neon.vst1x2.v16i8(i8* %a, <16 x i8> %1, <16 x i8> %2, i32 1)
  %tmp1 = getelementptr i8* %a, i32 32
  store i8* %tmp1, i8** %ptr
  ret void
}

define void @test_vst1x2_reg_update(i16* %a, [2 x <8 x i16>] %b.coerce, i16** %ptr, i32 %inc) #2 {
; CHECK: test_vst1x2_reg_update
; CHECK: st1 { v{{[0-9]+}}.8h, v{{[0-9]+}}.8h }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [2 x <8 x i16>] %b.coerce, 0
  %2 = extractvalue [2 x <8 x i16>] %b.coerce, 1
  %3 = bitcast i16* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x2.v8i16(i8* %3, <8 x i16> %1, <8 x i16> %2, i32 2)
  %tmp1 = getelementptr i16* %a, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret void
}

define void @test_vst1x3_fx_update(i32* %a, [3 x <2 x i32>] %b.coerce, i32** %ptr) #2 {
; CHECK: test_vst1x3_fx_update
; CHECK: st1 { v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s }, [x{{[0-9]+|sp}}], #24
  %1 = extractvalue [3 x <2 x i32>] %b.coerce, 0
  %2 = extractvalue [3 x <2 x i32>] %b.coerce, 1
  %3 = extractvalue [3 x <2 x i32>] %b.coerce, 2
  %4 = bitcast i32* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v2i32(i8* %4, <2 x i32> %1, <2 x i32> %2, <2 x i32> %3, i32 4)
  %tmp1 = getelementptr i32* %a, i32 6
  store i32* %tmp1, i32** %ptr
  ret void
}

define void @test_vst1x3_reg_update(i64* %a, [3 x <1 x i64>] %b.coerce, i64** %ptr, i32 %inc) #2 {
; CHECK: test_vst1x3_reg_update
; CHECK: st1 { v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [3 x <1 x i64>] %b.coerce, 0
  %2 = extractvalue [3 x <1 x i64>] %b.coerce, 1
  %3 = extractvalue [3 x <1 x i64>] %b.coerce, 2
  %4 = bitcast i64* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x3.v1i64(i8* %4, <1 x i64> %1, <1 x i64> %2, <1 x i64> %3, i32 8)
  %tmp1 = getelementptr i64* %a, i32 %inc
  store i64* %tmp1, i64** %ptr
  ret void
}

define void @test_vst1x4_fx_update(float* %a, [4 x <4 x float>] %b.coerce, float** %ptr) #2 {
; CHECK: test_vst1x4_fx_update
; CHECK: st1 { v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s }, [x{{[0-9]+|sp}}], #64
  %1 = extractvalue [4 x <4 x float>] %b.coerce, 0
  %2 = extractvalue [4 x <4 x float>] %b.coerce, 1
  %3 = extractvalue [4 x <4 x float>] %b.coerce, 2
  %4 = extractvalue [4 x <4 x float>] %b.coerce, 3
  %5 = bitcast float* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v4f32(i8* %5, <4 x float> %1, <4 x float> %2, <4 x float> %3, <4 x float> %4, i32 4)
  %tmp1 = getelementptr float* %a, i32 16
  store float* %tmp1, float** %ptr
  ret void
}

define void @test_vst1x4_reg_update(double* %a, [4 x <2 x double>] %b.coerce, double** %ptr, i32 %inc) #2 {
; CHECK: test_vst1x4_reg_update
; CHECK: st1 { v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [4 x <2 x double>] %b.coerce, 0
  %2 = extractvalue [4 x <2 x double>] %b.coerce, 1
  %3 = extractvalue [4 x <2 x double>] %b.coerce, 2
  %4 = extractvalue [4 x <2 x double>] %b.coerce, 3
  %5 = bitcast double* %a to i8*
  tail call void @llvm.aarch64.neon.vst1x4.v2f64(i8* %5, <2 x double> %1, <2 x double> %2, <2 x double> %3, <2 x double> %4, i32 8)
  %tmp1 = getelementptr double* %a, i32 %inc
  store double* %tmp1, double** %ptr
  ret void
}

declare { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.vld1x2.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x2.v8i16(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.vld1x3.v2i64(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.vld1x3.v8i16(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.vld1x4.v4f32(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.vld1x4.v8i8(i8*, i32)
declare void @llvm.aarch64.neon.vst1x2.v16i8(i8*, <16 x i8>, <16 x i8>, i32)
declare void @llvm.aarch64.neon.vst1x2.v8i16(i8*, <8 x i16>, <8 x i16>, i32)
declare void @llvm.aarch64.neon.vst1x3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.aarch64.neon.vst1x3.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.aarch64.neon.vst1x4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32) #3
declare void @llvm.aarch64.neon.vst1x4.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32) #3
