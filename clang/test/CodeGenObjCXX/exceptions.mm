// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fexceptions -o - %s | FileCheck %s

@interface OCType @end
void opaque();

namespace test0 {

  // CHECK: define void @_ZN5test03fooEv
  void foo() {
    try {
      // CHECK: invoke void @_Z6opaquev
      opaque();
    } catch (OCType *T) {
      // CHECK: call i32 (i8*, i8*, ...)* @llvm.eh.selector({{.*}} @__objc_personality_v0 {{.*}} @"OBJC_EHTYPE_$_OCType"
    }
  }
}
