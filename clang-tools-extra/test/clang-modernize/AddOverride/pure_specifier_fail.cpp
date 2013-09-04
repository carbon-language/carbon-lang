// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -add-override %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

// Test that override isn't placed correctly after "pure overrides"
struct A {
  virtual A *clone() = 0;
};
struct B : A {
  virtual B *clone() { return new B(); }
};
struct C : B {
  virtual B *clone() = 0;
  // CHECK: struct C : B {
  // CHECK: virtual B *clone() override = 0;
};
struct D : C {
  virtual D *clone() { return new D(); }
};
