; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s
; arm64 has equivalents of these tests separately.

define { [2 x <16 x i8>] } @test_vld2q_dup_fx_update(i8* %a, i8** %ptr) {
; CHECK-LABEL: test_vld2q_dup_fx_update
; CHECK: ld2r  { v{{[0-9]+}}.16b, v{{[0-9]+}}.16b }, [x{{[0-9]+|sp}}], #2
  %1 = tail call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2lane.v16i8(i8* %a, <16 x i8> undef, <16 x i8> undef, i32 0, i32 1)
  %2 = extractvalue { <16 x i8>, <16 x i8> } %1, 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  %4 = extractvalue { <16 x i8>, <16 x i8> } %1, 1
  %5 = shufflevector <16 x i8> %4, <16 x i8> undef, <16 x i32> zeroinitializer
  %6 = insertvalue { [2 x <16 x i8>] } undef, <16 x i8> %3, 0, 0
  %7 = insertvalue { [2 x <16 x i8>] } %6, <16 x i8> %5, 0, 1
  %tmp1 = getelementptr i8* %a, i32 2
  store i8* %tmp1, i8** %ptr
  ret { [2 x <16 x i8>] } %7
}

define { [2 x <4 x i32>] } @test_vld2q_dup_reg_update(i32* %a, i32** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld2q_dup_reg_update
; CHECK: ld2r  { v{{[0-9]+}}.4s, v{{[0-9]+}}.4s }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8* %1, <4 x i32> undef, <4 x i32> undef, i32 0, i32 4)
  %3 = extractvalue { <4 x i32>, <4 x i32> } %2, 0
  %4 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> zeroinitializer
  %5 = extractvalue { <4 x i32>, <4 x i32> } %2, 1
  %6 = shufflevector <4 x i32> %5, <4 x i32> undef, <4 x i32> zeroinitializer
  %7 = insertvalue { [2 x <4 x i32>] } undef, <4 x i32> %4, 0, 0
  %8 = insertvalue { [2 x <4 x i32>] } %7, <4 x i32> %6, 0, 1
  %tmp1 = getelementptr i32* %a, i32 %inc
  store i32* %tmp1, i32** %ptr
  ret { [2 x <4 x i32>] } %8
}

define { [3 x <4 x i16>] } @test_vld3_dup_fx_update(i16* %a, i16** %ptr) {
; CHECK-LABEL: test_vld3_dup_fx_update
; CHECK: ld3r  { v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h }, [x{{[0-9]+|sp}}], #6
  %1 = bitcast i16* %a to i8*
  %2 = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8* %1, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
  %3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 0
  %4 = shufflevector <4 x i16> %3, <4 x i16> undef, <4 x i32> zeroinitializer
  %5 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 1
  %6 = shufflevector <4 x i16> %5, <4 x i16> undef, <4 x i32> zeroinitializer
  %7 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %2, 2
  %8 = shufflevector <4 x i16> %7, <4 x i16> undef, <4 x i32> zeroinitializer
  %9 = insertvalue { [3 x <4 x i16>] }  undef, <4 x i16> %4, 0, 0
  %10 = insertvalue { [3 x <4 x i16>] }  %9, <4 x i16> %6, 0, 1
  %11 = insertvalue { [3 x <4 x i16>] }  %10, <4 x i16> %8, 0, 2
  %tmp1 = getelementptr i16* %a, i32 3
  store i16* %tmp1, i16** %ptr
  ret { [3 x <4 x i16>] }  %11
}

define { [3 x <8 x i8>] } @test_vld3_dup_reg_update(i8* %a, i8** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld3_dup_reg_update
; CHECK: ld3r  { v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3lane.v8i8(i8* %a, <8 x i8> undef, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 0
  %3 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %4 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 1
  %5 = shufflevector <8 x i8> %4, <8 x i8> undef, <8 x i32> zeroinitializer
  %6 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %1, 2
  %7 = shufflevector <8 x i8> %6, <8 x i8> undef, <8 x i32> zeroinitializer
  %8 = insertvalue { [3 x <8 x i8>] } undef, <8 x i8> %3, 0, 0
  %9 = insertvalue { [3 x <8 x i8>] } %8, <8 x i8> %5, 0, 1
  %10 = insertvalue { [3 x <8 x i8>] } %9, <8 x i8> %7, 0, 2
  %tmp1 = getelementptr i8* %a, i32 %inc
  store i8* %tmp1, i8** %ptr
  ret { [3 x <8 x i8>] }%10
}

define { [4 x <2 x i32>] } @test_vld4_dup_fx_update(i32* %a, i32** %ptr) #0 {
; CHECK-LABEL: test_vld4_dup_fx_update
; CHECK: ld4r  { v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s }, [x{{[0-9]+|sp}}], #16
  %1 = bitcast i32* %a to i8*
  %2 = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8* %1, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 0, i32 4)
  %3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 0
  %4 = shufflevector <2 x i32> %3, <2 x i32> undef, <2 x i32> zeroinitializer
  %5 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 1
  %6 = shufflevector <2 x i32> %5, <2 x i32> undef, <2 x i32> zeroinitializer
  %7 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 2
  %8 = shufflevector <2 x i32> %7, <2 x i32> undef, <2 x i32> zeroinitializer
  %9 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %2, 3
  %10 = shufflevector <2 x i32> %9, <2 x i32> undef, <2 x i32> zeroinitializer
  %11 = insertvalue { [4 x <2 x i32>] } undef, <2 x i32> %4, 0, 0
  %12 = insertvalue { [4 x <2 x i32>] } %11, <2 x i32> %6, 0, 1
  %13 = insertvalue { [4 x <2 x i32>] } %12, <2 x i32> %8, 0, 2
  %14 = insertvalue { [4 x <2 x i32>] } %13, <2 x i32> %10, 0, 3
  %tmp1 = getelementptr i32* %a, i32 4
  store i32* %tmp1, i32** %ptr
  ret { [4 x <2 x i32>] } %14
}

define { [4 x <2 x double>] } @test_vld4_dup_reg_update(double* %a, double** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld4_dup_reg_update
; CHECK: ld4r  { v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d }, [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = bitcast double* %a to i8*
  %2 = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8* %1, <2 x double> undef, <2 x double> undef, <2 x double> undef, <2 x double> undef, i32 0, i32 8)
  %3 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 0
  %4 = shufflevector <2 x double> %3, <2 x double> undef, <2 x i32> zeroinitializer
  %5 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 1
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <2 x i32> zeroinitializer
  %7 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 2
  %8 = shufflevector <2 x double> %7, <2 x double> undef, <2 x i32> zeroinitializer
  %9 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %2, 3
  %10 = shufflevector <2 x double> %9, <2 x double> undef, <2 x i32> zeroinitializer
  %11 = insertvalue { [4 x <2 x double>] } undef, <2 x double> %4, 0, 0
  %12 = insertvalue { [4 x <2 x double>] } %11, <2 x double> %6, 0, 1
  %13 = insertvalue { [4 x <2 x double>] } %12, <2 x double> %8, 0, 2
  %14 = insertvalue { [4 x <2 x double>] } %13, <2 x double> %10, 0, 3
  %tmp1 = getelementptr double* %a, i32 %inc
  store double* %tmp1, double** %ptr
  ret { [4 x <2 x double>] } %14
}

define { [2 x <8 x i8>] } @test_vld2_lane_fx_update(i8*  %a, [2 x <8 x i8>] %b, i8** %ptr) {
; CHECK-LABEL: test_vld2_lane_fx_update
; CHECK: ld2  { v{{[0-9]+}}.b, v{{[0-9]+}}.b }[7], [x{{[0-9]+|sp}}], #2
  %1 = extractvalue [2 x <8 x i8>] %b, 0
  %2 = extractvalue [2 x <8 x i8>] %b, 1
  %3 = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, i32 7, i32 1)
  %4 = extractvalue { <8 x i8>, <8 x i8> } %3, 0
  %5 = extractvalue { <8 x i8>, <8 x i8> } %3, 1
  %6 = insertvalue { [2 x <8 x i8>] } undef, <8 x i8> %4, 0, 0
  %7 = insertvalue { [2 x <8 x i8>] } %6, <8 x i8> %5, 0, 1
  %tmp1 = getelementptr i8* %a, i32 2
  store i8* %tmp1, i8** %ptr
  ret { [2 x <8 x i8>] } %7
}

define { [2 x <8 x i8>] } @test_vld2_lane_reg_update(i8*  %a, [2 x <8 x i8>] %b, i8** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld2_lane_reg_update
; CHECK: ld2  { v{{[0-9]+}}.b, v{{[0-9]+}}.b }[6], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [2 x <8 x i8>] %b, 0
  %2 = extractvalue [2 x <8 x i8>] %b, 1
  %3 = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, i32 6, i32 1)
  %4 = extractvalue { <8 x i8>, <8 x i8> } %3, 0
  %5 = extractvalue { <8 x i8>, <8 x i8> } %3, 1
  %6 = insertvalue { [2 x <8 x i8>] } undef, <8 x i8> %4, 0, 0
  %7 = insertvalue { [2 x <8 x i8>] } %6, <8 x i8> %5, 0, 1
  %tmp1 = getelementptr i8* %a, i32 %inc
  store i8* %tmp1, i8** %ptr
  ret { [2 x <8 x i8>] } %7
}

define { [3 x <2 x float>] } @test_vld3_lane_fx_update(float* %a, [3 x <2 x float>] %b, float** %ptr) {
; CHECK-LABEL: test_vld3_lane_fx_update
; CHECK: ld3  { v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s }[1], [x{{[0-9]+|sp}}], #12
  %1 = extractvalue [3 x <2 x float>] %b, 0
  %2 = extractvalue [3 x <2 x float>] %b, 1
  %3 = extractvalue [3 x <2 x float>] %b, 2
  %4 = bitcast float* %a to i8*
  %5 = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3lane.v2f32(i8* %4, <2 x float> %1, <2 x float> %2, <2 x float> %3, i32 1, i32 4)
  %6 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %5, 0
  %7 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %5, 1
  %8 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %5, 2
  %9 = insertvalue { [3 x <2 x float>] } undef, <2 x float> %6, 0, 0
  %10 = insertvalue { [3 x <2 x float>] } %9, <2 x float> %7, 0, 1
  %11 = insertvalue { [3 x <2 x float>] } %10, <2 x float> %8, 0, 2
  %tmp1 = getelementptr float* %a, i32 3
  store float* %tmp1, float** %ptr
  ret { [3 x <2 x float>] } %11
}

define { [3 x <4 x i16>] } @test_vld3_lane_reg_update(i16* %a, [3 x <4 x i16>] %b, i16** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld3_lane_reg_update
; CHECK: ld3  { v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h }[3], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [3 x <4 x i16>] %b, 0
  %2 = extractvalue [3 x <4 x i16>] %b, 1
  %3 = extractvalue [3 x <4 x i16>] %b, 2
  %4 = bitcast i16* %a to i8*
  %5 = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8* %4, <4 x i16> %1, <4 x i16> %2, <4 x i16> %3, i32 3, i32 2)
  %6 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %5, 0
  %7 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %5, 1
  %8 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %5, 2
  %9 = insertvalue { [3 x <4 x i16>] } undef, <4 x i16> %6, 0, 0
  %10 = insertvalue { [3 x <4 x i16>] } %9, <4 x i16> %7, 0, 1
  %11 = insertvalue { [3 x <4 x i16>] } %10, <4 x i16> %8, 0, 2
  %tmp1 = getelementptr i16* %a, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret { [3 x <4 x i16>] } %11
}

define { [4 x <2 x i32>] } @test_vld4_lane_fx_update(i32* readonly %a, [4 x <2 x i32>] %b, i32** %ptr) {
; CHECK-LABEL: test_vld4_lane_fx_update
; CHECK: ld4  { v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s }[1], [x{{[0-9]+|sp}}], #16
  %1 = extractvalue [4 x <2 x i32>] %b, 0
  %2 = extractvalue [4 x <2 x i32>] %b, 1
  %3 = extractvalue [4 x <2 x i32>] %b, 2
  %4 = extractvalue [4 x <2 x i32>] %b, 3
  %5 = bitcast i32* %a to i8*
  %6 = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8* %5, <2 x i32> %1, <2 x i32> %2, <2 x i32> %3, <2 x i32> %4, i32 1, i32 4)
  %7 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %6, 0
  %8 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %6, 1
  %9 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %6, 2
  %10 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %6, 3
  %11 = insertvalue { [4 x <2 x i32>] } undef, <2 x i32> %7, 0, 0
  %12 = insertvalue { [4 x <2 x i32>] } %11, <2 x i32> %8, 0, 1
  %13 = insertvalue { [4 x <2 x i32>] } %12, <2 x i32> %9, 0, 2
  %14 = insertvalue { [4 x <2 x i32>] } %13, <2 x i32> %10, 0, 3
  %tmp1 = getelementptr i32* %a, i32 4
  store i32* %tmp1, i32** %ptr
  ret { [4 x <2 x i32>] } %14
}

define { [4 x <2 x double>] } @test_vld4_lane_reg_update(double* readonly %a, [4 x <2 x double>] %b, double** %ptr, i32 %inc) {
; CHECK-LABEL: test_vld4_lane_reg_update
; CHECK: ld4  { v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d }[1], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [4 x <2 x double>] %b, 0
  %2 = extractvalue [4 x <2 x double>] %b, 1
  %3 = extractvalue [4 x <2 x double>] %b, 2
  %4 = extractvalue [4 x <2 x double>] %b, 3
  %5 = bitcast double* %a to i8*
  %6 = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8* %5, <2 x double> %1, <2 x double> %2, <2 x double> %3, <2 x double> %4, i32 1, i32 8)
  %7 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %6, 0
  %8 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %6, 1
  %9 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %6, 2
  %10 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %6, 3
  %11 = insertvalue { [4 x <2 x double>] } undef, <2 x double> %7, 0, 0
  %12 = insertvalue { [4 x <2 x double>] } %11, <2 x double> %8, 0, 1
  %13 = insertvalue { [4 x <2 x double>] } %12, <2 x double> %9, 0, 2
  %14 = insertvalue { [4 x <2 x double>] } %13, <2 x double> %10, 0, 3
  %tmp1 = getelementptr double* %a, i32 %inc
  store double* %tmp1, double** %ptr
  ret { [4 x <2 x double>] } %14
}

define void @test_vst2_lane_fx_update(i8* %a, [2 x <8 x i8>] %b, i8** %ptr) {
; CHECK-LABEL: test_vst2_lane_fx_update
; CHECK: st2  { v{{[0-9]+}}.b, v{{[0-9]+}}.b }[7], [x{{[0-9]+|sp}}], #2
  %1 = extractvalue [2 x <8 x i8>] %b, 0
  %2 = extractvalue [2 x <8 x i8>] %b, 1
  call void @llvm.arm.neon.vst2lane.v8i8(i8* %a, <8 x i8> %1, <8 x i8> %2, i32 7, i32 1)
  %tmp1 = getelementptr i8* %a, i32 2
  store i8* %tmp1, i8** %ptr
  ret void
}

define void @test_vst2_lane_reg_update(i32* %a, [2 x <2 x i32>] %b.coerce, i32** %ptr, i32 %inc) {
; CHECK-LABEL: test_vst2_lane_reg_update
; CHECK: st2  { v{{[0-9]+}}.s, v{{[0-9]+}}.s }[1], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [2 x <2 x i32>] %b.coerce, 0
  %2 = extractvalue [2 x <2 x i32>] %b.coerce, 1
  %3 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v2i32(i8* %3, <2 x i32> %1, <2 x i32> %2, i32 1, i32 4)
  %tmp1 = getelementptr i32* %a, i32 %inc
  store i32* %tmp1, i32** %ptr
  ret void
}

define void @test_vst3_lane_fx_update(float* %a, [3 x <4 x float>] %b, float** %ptr) {
; CHECK-LABEL: test_vst3_lane_fx_update
; CHECK: st3  { v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s }[3], [x{{[0-9]+|sp}}], #12
  %1 = extractvalue [3 x <4 x float>] %b, 0
  %2 = extractvalue [3 x <4 x float>] %b, 1
  %3 = extractvalue [3 x <4 x float>] %b, 2
  %4 = bitcast float* %a to i8*
  call void @llvm.arm.neon.vst3lane.v4f32(i8* %4, <4 x float> %1, <4 x float> %2, <4 x float> %3, i32 3, i32 4)
  %tmp1 = getelementptr float* %a, i32 3
  store float* %tmp1, float** %ptr
  ret void
}

; Function Attrs: nounwind
define void @test_vst3_lane_reg_update(i16* %a, [3 x <4 x i16>] %b, i16** %ptr, i32 %inc) {
; CHECK-LABEL: test_vst3_lane_reg_update
; CHECK: st3  { v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h }[3], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [3 x <4 x i16>] %b, 0
  %2 = extractvalue [3 x <4 x i16>] %b, 1
  %3 = extractvalue [3 x <4 x i16>] %b, 2
  %4 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v4i16(i8* %4, <4 x i16> %1, <4 x i16> %2, <4 x i16> %3, i32 3, i32 2)
  %tmp1 = getelementptr i16* %a, i32 %inc
  store i16* %tmp1, i16** %ptr
  ret void
}

define void @test_vst4_lane_fx_update(double* %a, [4 x <2 x double>] %b.coerce, double** %ptr) {
; CHECK-LABEL: test_vst4_lane_fx_update
; CHECK: st4  { v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d }[1], [x{{[0-9]+|sp}}], #32
  %1 = extractvalue [4 x <2 x double>] %b.coerce, 0
  %2 = extractvalue [4 x <2 x double>] %b.coerce, 1
  %3 = extractvalue [4 x <2 x double>] %b.coerce, 2
  %4 = extractvalue [4 x <2 x double>] %b.coerce, 3
  %5 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2f64(i8* %5, <2 x double> %1, <2 x double> %2, <2 x double> %3, <2 x double> %4, i32 1, i32 8)
  %tmp1 = getelementptr double* %a, i32 4
  store double* %tmp1, double** %ptr
  ret void
}


define void @test_vst4_lane_reg_update(float* %a, [4 x <2 x float>] %b.coerce, float** %ptr, i32 %inc) {
; CHECK-LABEL: test_vst4_lane_reg_update
; CHECK: st4  { v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s }[1], [x{{[0-9]+|sp}}], x{{[0-9]+}}
  %1 = extractvalue [4 x <2 x float>] %b.coerce, 0
  %2 = extractvalue [4 x <2 x float>] %b.coerce, 1
  %3 = extractvalue [4 x <2 x float>] %b.coerce, 2
  %4 = extractvalue [4 x <2 x float>] %b.coerce, 3
  %5 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2f32(i8* %5, <2 x float> %1, <2 x float> %2, <2 x float> %3, <2 x float> %4, i32 1, i32 4)
  %tmp1 = getelementptr float* %a, i32 %inc
  store float* %tmp1, float** %ptr
  ret void
}

declare { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32)
declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2lane.v16i8(i8*, <16 x i8>, <16 x i8>, i32, i32)
declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32)
declare { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32, i32)
