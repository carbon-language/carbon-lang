// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s

class B {
public:
  int bb;
  void fn2() {}
};

class A {
public:
  int aa;
  void fn1(B b) { b.fn2(); }
};

void foo(A *aptr) {
}

void bar() {
  A a;
}

// B should only be emitted as a forward reference.
// CHECK: metadata !"B", metadata !6, i32 3, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null} ; [ DW_TAG_class_type
