// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @b = external thread_local global
// CHECK: @d.e = internal thread_local global
// CHECK: @d.f = internal thread_local global
// CHECK: @a = thread_local global
__thread int a;
extern __thread int b;
int c() { return *&b; }
int d() {
  __thread static int e;
  __thread static union {float a; int b;} f = {.b = 1};
  return 0;
}

