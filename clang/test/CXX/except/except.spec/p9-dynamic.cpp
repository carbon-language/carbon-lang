// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() throw(int)
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK: %eh.selector = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 2, i8* bitcast (i8** @_ZTIi to i8*), i8* null) nounwind
// CHECK: ehspec.unexpected:
// CHECK: call void @__cxa_call_unexpected(i8* %1) noreturn
