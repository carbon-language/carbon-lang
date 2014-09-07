// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-mingw32 -fms-extensions -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test1
int test1(int *a, int i) {
// CHECK: %0 = load i32** %a.addr
// CHECK: %cmp = icmp ne i32* %0, null
// CHECK: call void @llvm.assume(i1 %cmp)
#ifdef _MSC_VER
  __assume(a != 0)
#else
  __builtin_assume(a != 0);
#endif

// Nothing is generated for an assume with side effects...
// CHECK-NOT: load i32** %i.addr
// CHECK-NOT: call void @llvm.assume
#ifdef _MSC_VER
  __assume(++i != 0)
#else
  __builtin_assume(++i != 0);
#endif

  return a[0];
}

