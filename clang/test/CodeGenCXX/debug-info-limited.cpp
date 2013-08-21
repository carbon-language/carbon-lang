// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// CHECK: ; [ DW_TAG_class_type ] [A] {{.*}} [def]
class A {
public:
  int z;
};

A *foo (A* x) {
  A *a = new A(*x);
  return a;
}

// Verify that we're not emitting a full definition of B in limit debug mode.
// CHECK: ; [ DW_TAG_class_type ] [B] {{.*}} [decl]

class B {
public:
  int y;
};

extern int bar(B *b);
int baz(B *b) {
  return bar(b);
}


// CHECK: ; [ DW_TAG_structure_type ] [C] {{.*}} [decl]

struct C {
};

C (*x)(C);
