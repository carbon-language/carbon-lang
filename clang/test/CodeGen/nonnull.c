// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm < %s | FileCheck %s

// CHECK: define void @foo(i32* nonnull %x)
void foo(int * __attribute__((nonnull)) x) {
  *x = 0;
}

// CHECK: define void @bar(i32* nonnull %x)
void bar(int * x) __attribute__((nonnull(1)))  {
  *x = 0;
}

// CHECK: define void @bar2(i32* %x, i32* nonnull %y)
void bar2(int * x, int * y) __attribute__((nonnull(2)))  {
  *x = 0;
}

static int a;
// CHECK: define nonnull i32* @bar3()
int * bar3() __attribute__((returns_nonnull))  {
  return &a;
}

