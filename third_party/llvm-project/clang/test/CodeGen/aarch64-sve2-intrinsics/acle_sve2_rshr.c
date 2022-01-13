// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svrshr_n_s8_z(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_z,)(pg, op1, 1);
}

svint8_t test_svrshr_n_s8_z_1(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_z_1
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_z,)(pg, op1, 8);
}

svint16_t test_svrshr_n_s16_z(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_z,)(pg, op1, 1);
}

svint16_t test_svrshr_n_s16_z_1(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_z,)(pg, op1, 16);
}

svint32_t test_svrshr_n_s32_z(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_z,)(pg, op1, 1);
}

svint32_t test_svrshr_n_s32_z_1(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_z,)(pg, op1, 32);
}

svint64_t test_svrshr_n_s64_z(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_z,)(pg, op1, 1);
}

svint64_t test_svrshr_n_s64_z_1(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_z,)(pg, op1, 64);
}

svuint8_t test_svrshr_n_u8_z(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_z,)(pg, op1, 1);
}

svuint8_t test_svrshr_n_u8_z_1(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_z_1
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_z,)(pg, op1, 8);
}

svuint16_t test_svrshr_n_u16_z(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_z,)(pg, op1, 1);
}

svuint16_t test_svrshr_n_u16_z_1(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_z,)(pg, op1, 16);
}

svuint32_t test_svrshr_n_u32_z(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_z,)(pg, op1, 1);
}

svuint32_t test_svrshr_n_u32_z_1(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_z,)(pg, op1, 32);
}

svuint64_t test_svrshr_n_u64_z(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_z,)(pg, op1, 1);
}

svuint64_t test_svrshr_n_u64_z_1(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_z_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_z'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_z,)(pg, op1, 64);
}

svint8_t test_svrshr_n_s8_m(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_m,)(pg, op1, 1);
}

svint8_t test_svrshr_n_s8_m_1(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_m_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_m,)(pg, op1, 8);
}

svint16_t test_svrshr_n_s16_m(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_m,)(pg, op1, 1);
}

svint16_t test_svrshr_n_s16_m_1(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_m,)(pg, op1, 16);
}

svint32_t test_svrshr_n_s32_m(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_m,)(pg, op1, 1);
}

svint32_t test_svrshr_n_s32_m_1(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_m,)(pg, op1, 32);
}

svint64_t test_svrshr_n_s64_m(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_m,)(pg, op1, 1);
}

svint64_t test_svrshr_n_s64_m_1(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_m,)(pg, op1, 64);
}

svuint8_t test_svrshr_n_u8_m(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_m,)(pg, op1, 1);
}

svuint8_t test_svrshr_n_u8_m_1(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_m_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_m,)(pg, op1, 8);
}

svuint16_t test_svrshr_n_u16_m(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_m,)(pg, op1, 1);
}

svuint16_t test_svrshr_n_u16_m_1(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_m,)(pg, op1, 16);
}

svuint32_t test_svrshr_n_u32_m(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_m,)(pg, op1, 1);
}

svuint32_t test_svrshr_n_u32_m_1(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_m,)(pg, op1, 32);
}

svuint64_t test_svrshr_n_u64_m(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_m,)(pg, op1, 1);
}

svuint64_t test_svrshr_n_u64_m_1(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_m_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_m'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_m,)(pg, op1, 64);
}

svint8_t test_svrshr_n_s8_x(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_x,)(pg, op1, 1);
}

svint8_t test_svrshr_n_s8_x_1(svbool_t pg, svint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s8_x_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s8_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s8,_x,)(pg, op1, 8);
}

svint16_t test_svrshr_n_s16_x(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_x,)(pg, op1, 1);
}

svint16_t test_svrshr_n_s16_x_1(svbool_t pg, svint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s16_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.srshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s16_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s16,_x,)(pg, op1, 16);
}

svint32_t test_svrshr_n_s32_x(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_x,)(pg, op1, 1);
}

svint32_t test_svrshr_n_s32_x_1(svbool_t pg, svint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s32_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s32_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s32,_x,)(pg, op1, 32);
}

svint64_t test_svrshr_n_s64_x(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_x,)(pg, op1, 1);
}

svint64_t test_svrshr_n_s64_x_1(svbool_t pg, svint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_s64_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.srshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_s64_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_s64,_x,)(pg, op1, 64);
}

svuint8_t test_svrshr_n_u8_x(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_x,)(pg, op1, 1);
}

svuint8_t test_svrshr_n_u8_x_1(svbool_t pg, svuint8_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u8_x_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urshr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u8_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u8,_x,)(pg, op1, 8);
}

svuint16_t test_svrshr_n_u16_x(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_x,)(pg, op1, 1);
}

svuint16_t test_svrshr_n_u16_x_1(svbool_t pg, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u16_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.urshr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u16_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u16,_x,)(pg, op1, 16);
}

svuint32_t test_svrshr_n_u32_x(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_x,)(pg, op1, 1);
}

svuint32_t test_svrshr_n_u32_x_1(svbool_t pg, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u32_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u32_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u32,_x,)(pg, op1, 32);
}

svuint64_t test_svrshr_n_u64_x(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_x,)(pg, op1, 1);
}

svuint64_t test_svrshr_n_u64_x_1(svbool_t pg, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshr_n_u64_x_1
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.urshr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, i32 64)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshr_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshr_n_u64_x'}}
  return SVE_ACLE_FUNC(svrshr,_n_u64,_x,)(pg, op1, 64);
}
