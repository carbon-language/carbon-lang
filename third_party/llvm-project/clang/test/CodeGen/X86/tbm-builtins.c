// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +tbm -emit-llvm -o - | FileCheck %s

#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/tbm-intrinsics-fast-isel.ll

unsigned int test__bextri_u32(unsigned int a) {
  // CHECK-LABEL: test__bextri_u32
  // CHECK: call i32 @llvm.x86.tbm.bextri.u32(i32 %{{.*}}, i32 1)
  return __bextri_u32(a, 1);
}

#ifdef __x86_64__
unsigned long long test__bextri_u64(unsigned long long a) {
  // CHECK-LABEL: test__bextri_u64
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64(i64 %{{.*}}, i64 2)
  return __bextri_u64(a, 2);
}

unsigned long long test__bextri_u64_bigint(unsigned long long a) {
  // CHECK-LABEL: test__bextri_u64_bigint
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64(i64 %{{.*}}, i64 549755813887)
  return __bextri_u64(a, 0x7fffffffffLL);
}
#endif

unsigned int test__blcfill_u32(unsigned int a) {
  // CHECK-LABEL: test__blcfill_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK: %{{.*}} = and i32 %{{.*}}, [[TMP]]
  return __blcfill_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcfill_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcfill_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK: %{{.*}} = and i64 %{{.*}}, [[TMP]]
  return __blcfill_u64(a);
}
#endif

unsigned int test__blci_u32(unsigned int a) {
  // CHECK-LABEL: test__blci_u32
  // CHECK: [[TMP1:%.*]] = add i32 %{{.*}}, 1
  // CHECK: [[TMP2:%.*]] = xor i32 [[TMP1]], -1
  // CHECK: %{{.*}} = or i32 %{{.*}}, [[TMP2]]
  return __blci_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blci_u64(unsigned long long a) {
  // CHECK-LABEL: test__blci_u64
  // CHECK: [[TMP1:%.*]] = add i64 %{{.*}}, 1
  // CHECK: [[TMP2:%.*]] = xor i64 [[TMP1]], -1
  // CHECK: %{{.*}} = or i64 %{{.*}}, [[TMP2]]
  return __blci_u64(a);
}
#endif

unsigned int test__blcic_u32(unsigned int a) {
  // CHECK-LABEL: test__blcic_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i32 [[TMP1]], [[TMP2]]
  return __blcic_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcic_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcic_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i64 [[TMP1]], [[TMP2]]
  return __blcic_u64(a);
}
#endif

unsigned int test__blcmsk_u32(unsigned int a) {
  // CHECK-LABEL: test__blcmsk_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = xor i32 %{{.*}}, [[TMP]]
  return __blcmsk_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcmsk_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcmsk_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = xor i64 %{{.*}}, [[TMP]]
  return __blcmsk_u64(a);
}
#endif

unsigned int test__blcs_u32(unsigned int a) {
  // CHECK-LABEL: test__blcs_u32
  // CHECK: [[TMP:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 %{{.*}}, [[TMP]]
  return __blcs_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blcs_u64(unsigned long long a) {
  // CHECK-LABEL: test__blcs_u64
  // CHECK: [[TMP:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 %{{.*}}, [[TMP]]
  return __blcs_u64(a);
}
#endif

unsigned int test__blsfill_u32(unsigned int a) {
  // CHECK-LABEL: test__blsfill_u32
  // CHECK: [[TMP:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 %{{.*}}, [[TMP]]
  return __blsfill_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blsfill_u64(unsigned long long a) {
  // CHECK-LABEL: test__blsfill_u64
  // CHECK: [[TMP:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 %{{.*}}, [[TMP]]
  return __blsfill_u64(a);
}
#endif

unsigned int test__blsic_u32(unsigned int a) {
  // CHECK-LABEL: test__blsic_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 [[TMP1]], [[TMP2]]
  return __blsic_u32(a);
}

#ifdef __x86_64__
unsigned long long test__blsic_u64(unsigned long long a) {
  // CHECK-LABEL: test__blsic_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 [[TMP1]], [[TMP2]]
  return __blsic_u64(a);
}
#endif

unsigned int test__t1mskc_u32(unsigned int a) {
  // CHECK-LABEL: test__t1mskc_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i32 [[TMP1]], [[TMP2]]
  return __t1mskc_u32(a);
}

#ifdef __x86_64__
unsigned long long test__t1mskc_u64(unsigned long long a) {
  // CHECK-LABEL: test__t1mskc_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = add i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = or i64 [[TMP1]], [[TMP2]]
  return __t1mskc_u64(a);
}
#endif

unsigned int test__tzmsk_u32(unsigned int a) {
  // CHECK-LABEL: test__tzmsk_u32
  // CHECK: [[TMP1:%.*]] = xor i32 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i32 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i32 [[TMP1]], [[TMP2]]
  return __tzmsk_u32(a);
}

#ifdef __x86_64__
unsigned long long test__tzmsk_u64(unsigned long long a) {
  // CHECK-LABEL: test__tzmsk_u64
  // CHECK: [[TMP1:%.*]] = xor i64 %{{.*}}, -1
  // CHECK: [[TMP2:%.*]] = sub i64 %{{.*}}, 1
  // CHECK-NEXT: {{.*}} = and i64 [[TMP1]], [[TMP2]]
  return __tzmsk_u64(a);
}
#endif
