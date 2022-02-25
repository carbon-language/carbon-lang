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

svint8_t test_svmls_s8_z(svbool_t pg, svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svmls_s8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s8,_z,)(pg, op1, op2, op3);
}

svint16_t test_svmls_s16_z(svbool_t pg, svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmls_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s16,_z,)(pg, op1, op2, op3);
}

svint32_t test_svmls_s32_z(svbool_t pg, svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmls_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s32,_z,)(pg, op1, op2, op3);
}

svint64_t test_svmls_s64_z(svbool_t pg, svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svmls_s64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s64,_z,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_u8_z(svbool_t pg, svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svmls_u8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u8,_z,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_u16_z(svbool_t pg, svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmls_u16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u16,_z,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_u32_z(svbool_t pg, svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmls_u32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u32,_z,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_u64_z(svbool_t pg, svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svmls_u64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u64,_z,)(pg, op1, op2, op3);
}

svint8_t test_svmls_s8_m(svbool_t pg, svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svmls_s8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s8,_m,)(pg, op1, op2, op3);
}

svint16_t test_svmls_s16_m(svbool_t pg, svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmls_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s16,_m,)(pg, op1, op2, op3);
}

svint32_t test_svmls_s32_m(svbool_t pg, svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmls_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s32,_m,)(pg, op1, op2, op3);
}

svint64_t test_svmls_s64_m(svbool_t pg, svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svmls_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s64,_m,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_u8_m(svbool_t pg, svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svmls_u8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u8,_m,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_u16_m(svbool_t pg, svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmls_u16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u16,_m,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_u32_m(svbool_t pg, svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmls_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u32,_m,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_u64_m(svbool_t pg, svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svmls_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u64,_m,)(pg, op1, op2, op3);
}

svint8_t test_svmls_s8_x(svbool_t pg, svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svmls_s8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s8,_x,)(pg, op1, op2, op3);
}

svint16_t test_svmls_s16_x(svbool_t pg, svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmls_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s16,_x,)(pg, op1, op2, op3);
}

svint32_t test_svmls_s32_x(svbool_t pg, svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmls_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s32,_x,)(pg, op1, op2, op3);
}

svint64_t test_svmls_s64_x(svbool_t pg, svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svmls_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_s64,_x,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_u8_x(svbool_t pg, svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svmls_u8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u8,_x,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_u16_x(svbool_t pg, svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmls_u16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u16,_x,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_u32_x(svbool_t pg, svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmls_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u32,_x,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_u64_x(svbool_t pg, svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svmls_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_u64,_x,)(pg, op1, op2, op3);
}

svint8_t test_svmls_n_s8_z(svbool_t pg, svint8_t op1, svint8_t op2, int8_t op3)
{
  // CHECK-LABEL: test_svmls_n_s8_z
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s8,_z,)(pg, op1, op2, op3);
}

svint16_t test_svmls_n_s16_z(svbool_t pg, svint16_t op1, svint16_t op2, int16_t op3)
{
  // CHECK-LABEL: test_svmls_n_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s16,_z,)(pg, op1, op2, op3);
}

svint32_t test_svmls_n_s32_z(svbool_t pg, svint32_t op1, svint32_t op2, int32_t op3)
{
  // CHECK-LABEL: test_svmls_n_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s32,_z,)(pg, op1, op2, op3);
}

svint64_t test_svmls_n_s64_z(svbool_t pg, svint64_t op1, svint64_t op2, int64_t op3)
{
  // CHECK-LABEL: test_svmls_n_s64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s64,_z,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_n_u8_z(svbool_t pg, svuint8_t op1, svuint8_t op2, uint8_t op3)
{
  // CHECK-LABEL: test_svmls_n_u8_z
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u8,_z,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_n_u16_z(svbool_t pg, svuint16_t op1, svuint16_t op2, uint16_t op3)
{
  // CHECK-LABEL: test_svmls_n_u16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u16,_z,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_n_u32_z(svbool_t pg, svuint32_t op1, svuint32_t op2, uint32_t op3)
{
  // CHECK-LABEL: test_svmls_n_u32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u32,_z,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_n_u64_z(svbool_t pg, svuint64_t op1, svuint64_t op2, uint64_t op3)
{
  // CHECK-LABEL: test_svmls_n_u64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u64,_z,)(pg, op1, op2, op3);
}

svint8_t test_svmls_n_s8_m(svbool_t pg, svint8_t op1, svint8_t op2, int8_t op3)
{
  // CHECK-LABEL: test_svmls_n_s8_m
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s8,_m,)(pg, op1, op2, op3);
}

svint16_t test_svmls_n_s16_m(svbool_t pg, svint16_t op1, svint16_t op2, int16_t op3)
{
  // CHECK-LABEL: test_svmls_n_s16_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s16,_m,)(pg, op1, op2, op3);
}

svint32_t test_svmls_n_s32_m(svbool_t pg, svint32_t op1, svint32_t op2, int32_t op3)
{
  // CHECK-LABEL: test_svmls_n_s32_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s32,_m,)(pg, op1, op2, op3);
}

svint64_t test_svmls_n_s64_m(svbool_t pg, svint64_t op1, svint64_t op2, int64_t op3)
{
  // CHECK-LABEL: test_svmls_n_s64_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s64,_m,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_n_u8_m(svbool_t pg, svuint8_t op1, svuint8_t op2, uint8_t op3)
{
  // CHECK-LABEL: test_svmls_n_u8_m
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u8,_m,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_n_u16_m(svbool_t pg, svuint16_t op1, svuint16_t op2, uint16_t op3)
{
  // CHECK-LABEL: test_svmls_n_u16_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u16,_m,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_n_u32_m(svbool_t pg, svuint32_t op1, svuint32_t op2, uint32_t op3)
{
  // CHECK-LABEL: test_svmls_n_u32_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u32,_m,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_n_u64_m(svbool_t pg, svuint64_t op1, svuint64_t op2, uint64_t op3)
{
  // CHECK-LABEL: test_svmls_n_u64_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u64,_m,)(pg, op1, op2, op3);
}

svint8_t test_svmls_n_s8_x(svbool_t pg, svint8_t op1, svint8_t op2, int8_t op3)
{
  // CHECK-LABEL: test_svmls_n_s8_x
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s8,_x,)(pg, op1, op2, op3);
}

svint16_t test_svmls_n_s16_x(svbool_t pg, svint16_t op1, svint16_t op2, int16_t op3)
{
  // CHECK-LABEL: test_svmls_n_s16_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s16,_x,)(pg, op1, op2, op3);
}

svint32_t test_svmls_n_s32_x(svbool_t pg, svint32_t op1, svint32_t op2, int32_t op3)
{
  // CHECK-LABEL: test_svmls_n_s32_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s32,_x,)(pg, op1, op2, op3);
}

svint64_t test_svmls_n_s64_x(svbool_t pg, svint64_t op1, svint64_t op2, int64_t op3)
{
  // CHECK-LABEL: test_svmls_n_s64_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_s64,_x,)(pg, op1, op2, op3);
}

svuint8_t test_svmls_n_u8_x(svbool_t pg, svuint8_t op1, svuint8_t op2, uint8_t op3)
{
  // CHECK-LABEL: test_svmls_n_u8_x
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u8,_x,)(pg, op1, op2, op3);
}

svuint16_t test_svmls_n_u16_x(svbool_t pg, svuint16_t op1, svuint16_t op2, uint16_t op3)
{
  // CHECK-LABEL: test_svmls_n_u16_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u16,_x,)(pg, op1, op2, op3);
}

svuint32_t test_svmls_n_u32_x(svbool_t pg, svuint32_t op1, svuint32_t op2, uint32_t op3)
{
  // CHECK-LABEL: test_svmls_n_u32_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u32,_x,)(pg, op1, op2, op3);
}

svuint64_t test_svmls_n_u64_x(svbool_t pg, svuint64_t op1, svuint64_t op2, uint64_t op3)
{
  // CHECK-LABEL: test_svmls_n_u64_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_u64,_x,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_f16_z(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmls_f16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %op3)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f16,_z,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svmls_f32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.sel.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %[[SEL]], <vscale x 4 x float> %op2, <vscale x 4 x float> %op3)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f32,_z,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_f64_z(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svmls_f64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.sel.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %[[SEL]], <vscale x 2 x double> %op2, <vscale x 2 x double> %op3)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f64,_z,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmls_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f16,_m,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svmls_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f32,_m,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svmls_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f64,_m,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmls_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f16,_x,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svmls_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f32,_x,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svmls_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_f64,_x,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_n_f16_z(svbool_t pg, svfloat16_t op1, svfloat16_t op2, float16_t op3)
{
  // CHECK-LABEL: test_svmls_n_f16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %[[DUP]])
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f16,_z,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_n_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2, float32_t op3)
{
  // CHECK-LABEL: test_svmls_n_f32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.sel.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %[[SEL]], <vscale x 4 x float> %op2, <vscale x 4 x float> %[[DUP]])
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f32,_z,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_n_f64_z(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
{
  // CHECK-LABEL: test_svmls_n_f64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %op3)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.sel.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %[[SEL]], <vscale x 2 x double> %op2, <vscale x 2 x double> %[[DUP]])
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f64,_z,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_n_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2, float16_t op3)
{
  // CHECK-LABEL: test_svmls_n_f16_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %[[DUP]])
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f16,_m,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_n_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2, float32_t op3)
{
  // CHECK-LABEL: test_svmls_n_f32_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %[[DUP]])
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f32,_m,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_n_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
{
  // CHECK-LABEL: test_svmls_n_f64_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %[[DUP]])
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f64,_m,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_n_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2, float16_t op3)
{
  // CHECK-LABEL: test_svmls_n_f16_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %[[DUP]])
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f16,_x,)(pg, op1, op2, op3);
}

svfloat32_t test_svmls_n_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2, float32_t op3)
{
  // CHECK-LABEL: test_svmls_n_f32_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %[[DUP]])
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f32,_x,)(pg, op1, op2, op3);
}

svfloat64_t test_svmls_n_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
{
  // CHECK-LABEL: test_svmls_n_f64_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %[[DUP]])
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls,_n_f64,_x,)(pg, op1, op2, op3);
}

svfloat16_t test_svmls_lane_f16(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.lane.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 0)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f16,,)(op1, op2, op3, 0);
}

svfloat16_t test_svmls_lane_f16_1(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.lane.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 7)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f16,,)(op1, op2, op3, 7);
}

svfloat32_t test_svmls_lane_f32(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 0)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f32,,)(op1, op2, op3, 0);
}

svfloat32_t test_svmls_lane_f32_1(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 3)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f32,,)(op1, op2, op3, 3);
}

svfloat64_t test_svmls_lane_f64(svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.lane.nxv2f64(<vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3, i32 0)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f64,,)(op1, op2, op3, 0);
}

svfloat64_t test_svmls_lane_f64_1(svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svmls_lane_f64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.lane.nxv2f64(<vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3, i32 1)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svmls_lane,_f64,,)(op1, op2, op3, 1);
}
