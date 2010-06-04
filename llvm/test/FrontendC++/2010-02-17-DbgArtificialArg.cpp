// RUN: %llvmgcc -g -S %s -o - | FileCheck %s
// Here, second to last argument "i32 64" indicates that artificial type is set.                                               
// Test to artificial attribute attahed to "this" pointer type.
// Radar 7655792 and 7655002

class A {
public:
  int fn1(int i) const { return i + 2; };
};

int foo() {
  A a;
  // Matching "i32 64, metadata !<number>} ; [ DW_TAG_pointer_type ]"
  // CHECK: i32 64, metadata {{![0-9]+\} ; \[ DW_TAG_pointer_type \]}}
  return a.fn1(1);
}
