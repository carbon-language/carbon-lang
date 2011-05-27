// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() throw(int)
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK: call i32 (i8*, i8*, ...)* @llvm.eh.selector({{.*}} i8* bitcast (i8** @_ZTIi to i8*), i8* null) nounwind
// CHECK: call void @__cxa_call_unexpected
