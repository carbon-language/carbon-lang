// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// CHECK: metadata !"_ZN1A3fooEv", {{.*}}, i32 258
class A {
protected:
  int foo();
}; 
A a;
