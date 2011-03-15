// RUN: %clang_cc1 %s -std=c++0x -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() noexcept
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK:  call i32 (i8*, i8*, ...)* @llvm.eh.selector({{.*}} i8* null) nounwind
// CHECK-NEXT: call void @_ZSt9terminatev() noreturn nounwind
// CHECK-NEXT: unreachable

void reverse() noexcept(false)
{
  // CHECK: call void @_Z8externalv()
  external();
}
