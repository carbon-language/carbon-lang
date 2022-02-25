// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svget2_s8(svint8x2_t tuple)
{
  // CHECK-LABEL: test_svget2_s8
  // CHECK: %[[EXT:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_s8,,)(tuple, 0);
}

svint16_t test_svget2_s16(svint16x2_t tuple)
{
  // CHECK-LABEL: test_svget2_s16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_s16,,)(tuple, 1);
}

svint32_t test_svget2_s32(svint32x2_t tuple)
{
  // CHECK-LABEL: test_svget2_s32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_s32,,)(tuple, 0);
}

svint64_t test_svget2_s64(svint64x2_t tuple)
{
  // CHECK-LABEL: test_svget2_s64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_s64,,)(tuple, 1);
}

svuint8_t test_svget2_u8(svuint8x2_t tuple)
{
  // CHECK-LABEL: test_svget2_u8
  // CHECK: %[[EXT:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_u8,,)(tuple, 0);
}

svuint16_t test_svget2_u16(svuint16x2_t tuple)
{
  // CHECK-LABEL: test_svget2_u16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_u16,,)(tuple, 1);
}

svuint32_t test_svget2_u32(svuint32x2_t tuple)
{
  // CHECK-LABEL: test_svget2_u32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_u32,,)(tuple, 0);
}

svuint64_t test_svget2_u64(svuint64x2_t tuple)
{
  // CHECK-LABEL: test_svget2_u64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_u64,,)(tuple, 1);
}

svfloat16_t test_svget2_f16(svfloat16x2_t tuple)
{
  // CHECK-LABEL: test_svget2_f16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 8 x half> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_f16,,)(tuple, 0);
}

svfloat32_t test_svget2_f32(svfloat32x2_t tuple)
{
  // CHECK-LABEL: test_svget2_f32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 4 x float> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_f32,,)(tuple, 1);
}

svfloat64_t test_svget2_f64(svfloat64x2_t tuple)
{
  // CHECK-LABEL: test_svget2_f64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 2 x double> %[[EXT]]
  return SVE_ACLE_FUNC(svget2,_f64,,)(tuple, 0);
}
