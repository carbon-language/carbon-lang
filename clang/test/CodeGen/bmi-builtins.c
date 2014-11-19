// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +bmi -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

// The double underscore intrinsics are for compatibility with 
// AMD's BMI interface. The single underscore intrinsics
// are for compatibility with Intel's BMI interface.
// Apart from the underscores, the interfaces are identical
// except in one case: although the 'bextr' register-form 
// instruction is identical in hardware, the AMD and Intel 
// intrinsics are different! 

unsigned short test__tzcnt_u16(unsigned short __X) {
  // CHECK: @llvm.cttz.i16
  return __tzcnt_u16(__X);
}

unsigned int test__andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: [[DEST:%.*]] = xor i32 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i32 %{{.*}}, [[DEST]]
  return __andn_u32(__X, __Y);
}

unsigned int test__bextr_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.bextr.32
  return __bextr_u32(__X, __Y);
}

unsigned int test__blsi_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = sub i32 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i32 [[SRC]], [[DEST]]
  return __blsi_u32(__X);
}

unsigned int test__blsmsk_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i32 [[DEST]], [[SRC]]
  return __blsmsk_u32(__X);
}

unsigned int test__blsr_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i32 [[DEST]], [[SRC]]
  return __blsr_u32(__X);
}

unsigned int test__tzcnt_u32(unsigned int __X) {
  // CHECK: @llvm.cttz.i32
  return __tzcnt_u32(__X);
}

unsigned long long test__andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: [[DEST:%.*]] = xor i64 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i64 %{{.*}}, [[DEST]]
  return __andn_u64(__X, __Y);
}

unsigned long long test__bextr_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: @llvm.x86.bmi.bextr.64
  return __bextr_u64(__X, __Y);
}

unsigned long long test__blsi_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = sub i64 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i64 [[SRC]], [[DEST]]
  return __blsi_u64(__X);
}

unsigned long long test__blsmsk_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i64 [[DEST]], [[SRC]]
  return __blsmsk_u64(__X);
}

unsigned long long test__blsr_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i64 [[DEST]], [[SRC]]
  return __blsr_u64(__X);
}

unsigned long long test__tzcnt_u64(unsigned long long __X) {
  // CHECK: @llvm.cttz.i64
  return __tzcnt_u64(__X);
}

// Intel intrinsics

unsigned short test_tzcnt_u16(unsigned short __X) {
  // CHECK: @llvm.cttz.i16
  return _tzcnt_u16(__X);
}

unsigned int test_andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: [[DEST:%.*]] = xor i32 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i32 %{{.*}}, [[DEST]]
  return _andn_u32(__X, __Y);
}

unsigned int test_bextr_u32(unsigned int __X, unsigned int __Y, 
                            unsigned int __Z) {
  // CHECK: @llvm.x86.bmi.bextr.32
  return _bextr_u32(__X, __Y, __Z);
}

unsigned int test_blsi_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = sub i32 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i32 [[SRC]], [[DEST]]
  return _blsi_u32(__X);
}

unsigned int test_blsmsk_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i32 [[DEST]], [[SRC]]
  return _blsmsk_u32(__X);
}

unsigned int test_blsr_u32(unsigned int __X) {
  // CHECK: [[DEST:%.*]] = add i32 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i32 [[DEST]], [[SRC]]
  return _blsr_u32(__X);
}

unsigned int test_tzcnt_u32(unsigned int __X) {
  // CHECK: @llvm.cttz.i32
  return _tzcnt_u32(__X);
}

unsigned long long test_andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: [[DEST:%.*]] = xor i64 %{{.*}}, -1
  // CHECK-NEXT: %{{.*}} = and i64 %{{.*}}, [[DEST]]
  return _andn_u64(__X, __Y);
}

unsigned long long test_bextr_u64(unsigned long __X, unsigned int __Y, 
                                  unsigned int __Z) {
  // CHECK: @llvm.x86.bmi.bextr.64
  return _bextr_u64(__X, __Y, __Z);
}

unsigned long long test_blsi_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = sub i64 0, [[SRC:%.*]]
  // CHECK-NEXT: %{{.*}} = and i64 [[SRC]], [[DEST]]
  return _blsi_u64(__X);
}

unsigned long long test_blsmsk_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = xor i64 [[DEST]], [[SRC]]
  return _blsmsk_u64(__X);
}

unsigned long long test_blsr_u64(unsigned long long __X) {
  // CHECK: [[DEST:%.*]] = add i64 [[SRC:%.*]], -1
  // CHECK-NEXT: %{{.*}} = and i64 [[DEST]], [[SRC]]
  return _blsr_u64(__X);
}

unsigned long long test_tzcnt_u64(unsigned long long __X) {
  // CHECK: @llvm.cttz.i64
  return _tzcnt_u64(__X);
}
