// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -o - | FileCheck %s
#include <typeinfo>
struct A { virtual void f(); };
struct B : A { };

// CHECK: {{define.*@_Z1fP1A}}
B fail;
const B& f(A *a) {
  try {
    // CHECK: call i8* @__dynamic_cast
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_cast() noreturn
    dynamic_cast<const B&>(*a);
  } catch (std::bad_cast&) {
    // CHECK: call i8* @llvm.eh.exception
    // CHECK: {{call.*llvm.eh.selector.*_ZTISt8bad_cast}}
    // CHECK: {{call i32 @llvm.eh.typeid.for.*@_ZTISt8bad_cast}}
  }
  return fail;
}
