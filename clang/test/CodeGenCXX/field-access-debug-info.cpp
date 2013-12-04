// RUN: %clang -g -S -emit-llvm %s -o - | FileCheck %s

// CHECK: [ DW_TAG_member ] [p] [{{[^]]*}}] [from int]
// CHECK: [ DW_TAG_member ] [pr] [{{[^]]*}}] [private] [from int]

class A {
public:
  int p;
private:
  int pr;
};

A a;
