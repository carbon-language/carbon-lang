// RUN: %clang_cc1 %s -std=c++0x -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() noexcept
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK: terminate.lpad:
// CHECK:  %eh.selector = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null) nounwind
// CHECK-NEXT: call void @_ZSt9terminatev() noreturn nounwind
// CHECK-NEXT: unreachable

void reverse() noexcept(false)
{
  // CHECK: call void @_Z8externalv()
  external();
}
