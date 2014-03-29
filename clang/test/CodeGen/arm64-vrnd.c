// RUN: %clang_cc1 -triple arm64-apple-ios7 -ffreestanding -emit-llvm -o - %s | FileCheck %s

#include <arm_neon.h>

int32x2_t rnd1(float32x2_t a) { return vrnd_f32(a); }
// CHECK: call <2 x float> @llvm.trunc.v2f32(<2 x float>
int32x4_t rnd3(float32x4_t a) { return vrndq_f32(a); }
// CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float>
int64x2_t rnd5(float64x2_t a) { return vrndq_f64(a); }
// CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double>


int32x2_t rnd7(float32x2_t a) { return vrndn_f32(a); }
// CHECK: call <2 x float> @llvm.arm64.neon.frintn.v2f32(<2 x float>
int32x4_t rnd8(float32x4_t a) { return vrndnq_f32(a); }
// CHECK: call <4 x float> @llvm.arm64.neon.frintn.v4f32(<4 x float>
int64x2_t rnd9(float64x2_t a) { return vrndnq_f64(a); }
// CHECK: call <2 x double> @llvm.arm64.neon.frintn.v2f64(<2 x double>
int64x2_t rnd10(float64x2_t a) { return vrndnq_f64(a); }
// CHECK: call <2 x double> @llvm.arm64.neon.frintn.v2f64(<2 x double>

int32x2_t rnd11(float32x2_t a) { return vrndm_f32(a); }
// CHECK: call <2 x float> @llvm.floor.v2f32(<2 x float>
int32x4_t rnd12(float32x4_t a) { return vrndmq_f32(a); }
// CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float>
int64x2_t rnd13(float64x2_t a) { return vrndmq_f64(a); }
// CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double>
int64x2_t rnd14(float64x2_t a) { return vrndmq_f64(a); }
// CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double>

int32x2_t rnd15(float32x2_t a) { return vrndp_f32(a); }
// CHECK: call <2 x float> @llvm.ceil.v2f32(<2 x float>
int32x4_t rnd16(float32x4_t a) { return vrndpq_f32(a); }
// CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float>
int64x2_t rnd18(float64x2_t a) { return vrndpq_f64(a); }
// CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double>

int32x2_t rnd19(float32x2_t a) { return vrnda_f32(a); }
// CHECK: call <2 x float> @llvm.round.v2f32(<2 x float>
int32x4_t rnd20(float32x4_t a) { return vrndaq_f32(a); }
// CHECK: call <4 x float> @llvm.round.v4f32(<4 x float>
int64x2_t rnd22(float64x2_t a) { return vrndaq_f64(a); }
// CHECK: call <2 x double> @llvm.round.v2f64(<2 x double>

int32x2_t rnd23(float32x2_t a) { return vrndx_f32(a); }
// CHECK: call <2 x float> @llvm.rint.v2f32(<2 x float>
int32x4_t rnd24(float32x4_t a) { return vrndxq_f32(a); }
// CHECK: call <4 x float> @llvm.rint.v4f32(<4 x float>
int64x2_t rnd25(float64x2_t a) { return vrndxq_f64(a); }
// CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double>

