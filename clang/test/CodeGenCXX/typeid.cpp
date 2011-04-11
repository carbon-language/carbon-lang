// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
#include <typeinfo>

// PR7400
struct A { virtual void f(); };

// CHECK: define i8* @_Z1fv
const char *f() {
  try {
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_typeid() noreturn
    return typeid(*static_cast<A *>(0)).name();
  } catch (...) {
    // CHECK: call i8* @llvm.eh.exception
  }

  return 0;
}
