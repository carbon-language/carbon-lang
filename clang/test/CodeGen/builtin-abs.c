// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int absi(int x) {
// CHECK-LABEL: @absi(
// CHECK:   [[NEG:%.*]] = sub i32 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp sge i32 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i32 [[X]], i32 [[NEG]]
//
  return __builtin_abs(x);
}

long absl(long x) {
// CHECK-LABEL: @absl(
// CHECK:   [[NEG:%.*]] = sub i64 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp sge i64 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i64 [[X]], i64 [[NEG]]
//
  return __builtin_labs(x);
}

long long absll(long long x) {
// CHECK-LABEL: @absll(
// CHECK:   [[NEG:%.*]] = sub i64 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp sge i64 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i64 [[X]], i64 [[NEG]]
//
  return __builtin_llabs(x);
}

