// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
#include <typeinfo>

namespace Test1 {

// PR7400
struct A { virtual void f(); };

// CHECK: define i8* @_ZN5Test11fEv
const char *f() {
  try {
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_typeid() noreturn
    return typeid(*static_cast<A *>(0)).name();
  } catch (...) {
    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK-NEXT:   catch i8* null
  }

  return 0;
}

}
