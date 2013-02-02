// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
enum D : short; // enum forward declaration
A a;
B b;
C c;
D d;

// CHECK: metadata !{i32 {{.*}}, null, metadata !"A", metadata ![[FILE:.*]], i32 3, i64 32, i64 32, i32 0, i32 0, metadata !{{.*}}, metadata !{{.*}}, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"B", metadata ![[FILE]], i32 4, i64 64, i64 64, i32 0, i32 0, metadata !{{.*}}, metadata !{{.*}}, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"C", metadata ![[FILE]], i32 5, i64 32, i64 32, i32 0, i32 0, null, metadata !{{.*}}, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"D", metadata ![[FILE]], i32 6, i64 16, i64 16, i32 0, i32 4, null, null, i32 0} ; [ DW_TAG_enumeration_type ]

namespace PR14029 {
  // Make sure this doesn't crash/assert.
  template <typename T> struct Test {
    enum class Tag {
      test = 0
    };
    Test() {
      auto t = Tag::test;
    }
    Tag tag() const { return static_cast<Tag>(1); }
  };
  Test<int> t;
}
