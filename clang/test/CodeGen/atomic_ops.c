// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

void foo(int x)
{
  _Atomic(int) i = 0;
  _Atomic(short) j = 0;
  // Check that multiply / divides on atomics produce a cmpxchg loop
  i *= 2;
  // CHECK: mul nsw i32
  // CHECK: {{(cmpxchg i32*|i1 @__atomic_compare_exchange\(i32 4,)}}
  i /= 2;
  // CHECK: sdiv i32
  // CHECK: {{(cmpxchg i32*|i1 @__atomic_compare_exchange\(i32 4, )}}
  j /= x;
  // CHECK: sdiv i32
  // CHECK: {{(cmpxchg i16*|i1 @__atomic_compare_exchange\(i32 2, )}}

}

extern _Atomic _Bool b;

_Bool bar() {
// CHECK-LABEL: @bar
// CHECK: %[[load:.*]] = load atomic i8* @b seq_cst, align 1
// CHECK: %[[tobool:.*]] = trunc i8 %[[load]] to i1
// CHECK: ret i1 %[[tobool]]
  return b;
}
