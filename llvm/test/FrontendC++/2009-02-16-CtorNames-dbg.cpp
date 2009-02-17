// RUN: %llvmgcc -S -g --emit-llvm %s -o - | grep "\~A"
// RUN: %llvmgcc -S -g --emit-llvm %s -o - | not grep comp_ctor
// RUN: %llvmgcc -S -g --emit-llvm %s -o - | not grep comp_dtor
class A {
  int i;
public:
  A() { i = 0; }
 ~A() { i = 42; }
};

A a;

