// RUN: %llvmgcc -S -g --emit-llvm %s -o - | grep "\~A"
// RUN: %llvmgcc -S -g --emit-llvm %s -o - | not grep comp_ctor
// RUN: %llvmgcc -S -g --emit-llvm %s -o - | not grep comp_dtor
// FIXME: This is failing on Darwin because of either r66861 or r66859.
// XFAIL: darwin
class A {
  int i;
public:
  A() { i = 0; }
 ~A() { i = 42; }
};

A a;

