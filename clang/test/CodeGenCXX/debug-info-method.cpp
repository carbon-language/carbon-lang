// RUN: %clang -fverbose-asm -g -S %s -o - | FileCheck %s
// CHECK: DW_ACCESS_protected
class A {
protected:
  int foo();
}; 
A a;
