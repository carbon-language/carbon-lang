// RUN: %clang_cc1 %s -triple=x86_64-unknown-unknown -target-feature +tbm -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned int test__bextri_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.bextri.u32
  return __bextri_u32(a, 1);
}

unsigned long long test__bextri_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64
  return __bextri_u64(a, 2);
}

unsigned long long test__bextri_u64_bigint(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.bextri.u64
  return __bextri_u64(a, 0x7fffffffffLL);
}

unsigned int test__blcfill_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blcfill.u32
  return __blcfill_u32(a);
}

unsigned long long test__blcfill_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blcfill.u64
  return __blcfill_u64(a);
}

unsigned int test__blci_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blci.u32
  return __blci_u32(a);
}

unsigned long long test__blci_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blci.u64
  return __blci_u64(a);
}

unsigned int test__blcic_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blcic.u32
  return __blcic_u32(a);
}

unsigned long long test__blcic_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blcic.u64
  return __blcic_u64(a);
}

unsigned int test__blcmsk_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blcmsk.u32
  return __blcmsk_u32(a);
}

unsigned long long test__blcmsk_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blcmsk.u64
  return __blcmsk_u64(a);
}

unsigned int test__blcs_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blcs.u32
  return __blcs_u32(a);
}

unsigned long long test__blcs_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blcs.u64
  return __blcs_u64(a);
}

unsigned int test__blsfill_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blsfill.u32
  return __blsfill_u32(a);
}

unsigned long long test__blsfill_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blsfill.u64
  return __blsfill_u64(a);
}

unsigned int test__blsic_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.blsic.u32
  return __blsic_u32(a);
}

unsigned long long test__blsic_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.blsic.u64
  return __blsic_u64(a);
}

unsigned int test__t1mskc_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.t1mskc.u32
  return __t1mskc_u32(a);
}

unsigned long long test__t1mskc_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.t1mskc.u64
  return __t1mskc_u64(a);
}

unsigned int test__tzmsk_u32(unsigned int a) {
  // CHECK: call i32 @llvm.x86.tbm.tzmsk.u32
  return __tzmsk_u32(a);
}

unsigned long long test__tzmsk_u64(unsigned long long a) {
  // CHECK: call i64 @llvm.x86.tbm.tzmsk.u64
  return __tzmsk_u64(a);
}
