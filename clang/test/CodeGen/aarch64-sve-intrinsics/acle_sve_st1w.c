// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svst1w_s64(svbool_t pg, int32_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i1> %[[PG]], i32* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w,_s64,,)(pg, base, data);
}

void test_svst1w_u64(svbool_t pg, uint32_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i1> %[[PG]], i32* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w,_u64,,)(pg, base, data);
}

void test_svst1w_vnum_s64(svbool_t pg, int32_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1w_vnum_u64(svbool_t pg, uint32_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_vnum,_u64,,)(pg, base, vnum, data);
}

void test_svst1w_scatter_u64base_s64(svbool_t pg, svuint64_t bases, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,,_s64)(pg, bases, data);
}

void test_svst1w_scatter_u64base_u64(svbool_t pg, svuint64_t bases, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,,_u64)(pg, bases, data);
}

void test_svst1w_scatter_s64offset_s64(svbool_t pg, int32_t *base, svint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_s64offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,s64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1w_scatter_s64offset_u64(svbool_t pg, uint32_t *base, svint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_s64offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,s64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1w_scatter_u64offset_s64(svbool_t pg, int32_t *base, svuint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,u64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1w_scatter_u64offset_u64(svbool_t pg, uint32_t *base, svuint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,u64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1w_scatter_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,_offset,_s64)(pg, bases, offset, data);
}

void test_svst1w_scatter_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,_offset,_u64)(pg, bases, offset, data);
}

void test_svst1w_scatter_s64index_s64(svbool_t pg, int32_t *base, svint64_t indices, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_s64index_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,s64,index,_s64)(pg, base, indices, data);
}

void test_svst1w_scatter_s64index_u64(svbool_t pg, uint32_t *base, svint64_t indices, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_s64index_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32> %0, <vscale x 2 x i1> %1, i32* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,s64,index,_u64)(pg, base, indices, data);
}

void test_svst1w_scatter_u64index_s64(svbool_t pg, int32_t *base, svuint64_t indices, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64index_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,u64,index,_s64)(pg, base, indices, data);
}

void test_svst1w_scatter_u64index_u64(svbool_t pg, uint32_t *base, svuint64_t indices, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64index_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i32* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter_,u64,index,_u64)(pg, base, indices, data);
}

void test_svst1w_scatter_u64base_index_s64(svbool_t pg, svuint64_t bases, int64_t index, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_index_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,_index,_s64)(pg, bases, index, data);
}

void test_svst1w_scatter_u64base_index_u64(svbool_t pg, svuint64_t bases, int64_t index, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_scatter_u64base_index_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_scatter,_u64base,_index,_u64)(pg, bases, index, data);
}
