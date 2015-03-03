// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// CHECK: !MDCompositeType(tag: DW_TAG_class_type, name: "A"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
class A {
public:
  int z;
};

A *foo (A* x) {
  A *a = new A(*x);
  return a;
}

// CHECK: !MDCompositeType(tag: DW_TAG_class_type, name: "B"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

class B {
public:
  int y;
};

extern int bar(B *b);
int baz(B *b) {
  return bar(b);
}


// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "C"
// CHECK-SAME:             flags: DIFlagFwdDecl

struct C {
};

C (*x)(C);
