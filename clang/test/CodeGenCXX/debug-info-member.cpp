// RUN: %clang -fverbose-asm -cc1 -g -S %s -o - | grep DW_ACCESS_public
class A {
public:
  int x;
}; 
A a;
