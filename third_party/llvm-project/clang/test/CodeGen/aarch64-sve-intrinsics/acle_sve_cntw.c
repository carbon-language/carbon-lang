// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

uint64_t test_svcntw()
{
  // CHECK-LABEL: test_svcntw
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.vscale.i64()
  // CHECK-NEXT: %[[RET:.*]] = shl i64 %[[INTRINSIC]], 2
  // CHECK: ret i64 %[[RET]]
  return svcntw();
}

uint64_t test_svcntw_pat()
{
  // CHECK-LABEL: test_svcntw_pat
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 0)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_POW2);
}

uint64_t test_svcntw_pat_1()
{
  // CHECK-LABEL: test_svcntw_pat_1
  // CHECK: ret i64 1
  return svcntw_pat(SV_VL1);
}

uint64_t test_svcntw_pat_2()
{
  // CHECK-LABEL: test_svcntw_pat_2
  // CHECK: ret i64 2
  return svcntw_pat(SV_VL2);
}

uint64_t test_svcntw_pat_3()
{
  // CHECK-LABEL: test_svcntw_pat_3
  // CHECK: ret i64 3
  return svcntw_pat(SV_VL3);
}

uint64_t test_svcntw_pat_4()
{
  // CHECK-LABEL: test_svcntw_pat_4
  // CHECK: ret i64 4
  return svcntw_pat(SV_VL4);
}

uint64_t test_svcntw_pat_5()
{
  // CHECK-LABEL: test_svcntw_pat_5
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 5)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL5);
}

uint64_t test_svcntw_pat_6()
{
  // CHECK-LABEL: test_svcntw_pat_6
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 6)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL6);
}

uint64_t test_svcntw_pat_7()
{
  // CHECK-LABEL: test_svcntw_pat_7
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 7)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL7);
}

uint64_t test_svcntw_pat_8()
{
  // CHECK-LABEL: test_svcntw_pat_8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 8)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL8);
}

uint64_t test_svcntw_pat_9()
{
  // CHECK-LABEL: test_svcntw_pat_9
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 9)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL16);
}

uint64_t test_svcntw_pat_10()
{
  // CHECK-LABEL: test_svcntw_pat_10
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 10)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL32);
}

uint64_t test_svcntw_pat_11()
{
  // CHECK-LABEL: test_svcntw_pat_11
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 11)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL64);
}

uint64_t test_svcntw_pat_12()
{
  // CHECK-LABEL: test_svcntw_pat_12
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 12)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL128);
}

uint64_t test_svcntw_pat_13()
{
  // CHECK-LABEL: test_svcntw_pat_13
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 13)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_VL256);
}

uint64_t test_svcntw_pat_14()
{
  // CHECK-LABEL: test_svcntw_pat_14
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 29)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_MUL4);
}

uint64_t test_svcntw_pat_15()
{
  // CHECK-LABEL: test_svcntw_pat_15
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntw(i32 30)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntw_pat(SV_MUL3);
}

uint64_t test_svcntw_pat_16()
{
  // CHECK-LABEL: test_svcntw_pat_16
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.vscale.i64()
  // CHECK-NEXT: %[[RET:.*]] = shl i64 %[[INTRINSIC]], 2
  // CHECK: ret i64 %[[RET]]
  return svcntw_pat(SV_ALL);
}
