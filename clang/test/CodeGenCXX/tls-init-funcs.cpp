// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8 -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

// CHECK: @a = internal thread_local global
// CHECK: @_tlv_atexit({{.*}}@_ZN1AD1Ev
// CHECK: define weak hidden {{.*}} @_ZTW1a

struct A {
  ~A();
};

thread_local A a;
