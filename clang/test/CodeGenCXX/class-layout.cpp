// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// An extra byte should be allocated for an empty class.
namespace Test1 {
  // CHECK: %"struct.Test1::A" = type { i8 }
  struct A { } *a;
}

namespace Test2 {
  // No need to add tail padding here.
  // CHECK: %"struct.Test2::A" = type { i8*, i32 }
  struct A { void *a; int b; } *a;
}

namespace Test3 {
  // C should have a vtable pointer.
  // CHECK: %"struct.Test3::A" = type { i32 (...)**, i32 }
  struct A { virtual void f(); int a; } *a;
}
