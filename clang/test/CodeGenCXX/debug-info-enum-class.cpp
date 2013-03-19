// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
enum D : short; // enum forward declaration
A a;
B b;
C c;
D d;

// CHECK: ; [ DW_TAG_enumeration_type ] [A] [line 3, size 32, align 32, offset 0] [from int]
// CHECK: ; [ DW_TAG_enumeration_type ] [B] [line 4, size 64, align 64, offset 0] [from long unsigned int]
// CHECK: ; [ DW_TAG_enumeration_type ] [C] [line 5, size 32, align 32, offset 0] [from ]
// CHECK: ; [ DW_TAG_enumeration_type ] [D] [line 6, size 16, align 16, offset 0] [fwd] [from ]

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
