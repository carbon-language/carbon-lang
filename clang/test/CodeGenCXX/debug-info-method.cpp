// RUN: %clang -fverbose-asm -cc1 -g -S %s -o - | grep DW_ACCESS_protected
class A {
protected:
  int foo();
}; 
A a;
