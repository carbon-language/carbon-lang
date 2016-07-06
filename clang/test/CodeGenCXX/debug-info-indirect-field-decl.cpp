// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s
//
// Test that indirect field decls are handled gracefully.
// rdar://problem/16348575
//
template <class T, int T::*ptr> class Foo {  };

struct Bar {
  int i1;
  // CHECK: ![[INT:[0-9]+]] = !DIBasicType(name: "int"
  // CHECK: ![[UNION:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_union_type,{{.*}} identifier: "_ZTSN3BarUt_E")
  union {
    // CHECK: !DIDerivedType(tag: DW_TAG_member, name: "i2",
    // CHECK-SAME:           line: [[@LINE+9]]
    // CHECK-SAME:           baseType: ![[INT]]
    // CHECK-SAME:           size: 32, align: 32
    // CHECK-NOT:            offset:
    // CHECK-SAME:           ){{$}}
    // CHECK: !DIDerivedType(tag: DW_TAG_member, scope:
    // CHECK-SAME:           line: [[@LINE-8]]
    // CHECK-SAME:           baseType: ![[UNION]]
    // CHECK-SAME:           size: 32, align: 32, offset: 32
    int i2;
  };
};

Foo<Bar, &Bar::i2> the_foo;
