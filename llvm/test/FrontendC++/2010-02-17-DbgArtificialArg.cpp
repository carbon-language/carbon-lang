// RUN: %llvmgcc -g -S %s -o - | grep DW_TAG_pointer_type |  grep "i32 458767, metadata .., metadata ..., metadata .., i32 0, i64 64, i64 64, i64 0, i32 64, metadata ..."
// Here, second to last argument "i32 64" indicates that artificial type is set.                                               
// Test to artificial attribute attahed to "this" pointer type.
// Radar 7655792 and 7655002

class A {
public:
  int fn1(int i) const { return i + 2; };
};

int foo() {
  A a;
  return a.fn1(1);
}
