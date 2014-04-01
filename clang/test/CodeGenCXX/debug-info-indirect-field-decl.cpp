// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
//
// Test that indirect field decls are handled gracefully.
// rdar://problem/16348575
//
template <class T, int T::*ptr> class Foo {  };

struct Bar {
  int i1;
  // CHECK: [ DW_TAG_member ] [line [[@LINE+1]], size 32, align 32, offset 32] [from _ZTSN3BarUt_E]
  union {
    // CHECK: [ DW_TAG_member ] [i2] [line [[@LINE+1]], size 32, align 32, offset 0] [from int]
    int i2;
  };
};

Foo<Bar, &Bar::i2> the_foo;
