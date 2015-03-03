// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
enum D : short; // enum forward declaration
A a;
B b;
C c;
D d;

// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "A"
// CHECK-SAME:             line: 3
// CHECK-SAME:             baseType: ![[INT:[0-9]+]]
// CHECK-SAME:             size: 32, align: 32
// CHECK-NOT:              offset:
// CHECK-NOT:              flags:
// CHECK-SAME:             ){{$}}
// CHECK: ![[INT]] = !MDBasicType(name: "int"
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "B"
// CHECK-SAME:             line: 4
// CHECK-SAME:             baseType: ![[ULONG:[0-9]+]]
// CHECK-SAME:             size: 64, align: 64
// CHECK-NOT:              offset:
// CHECK-NOT:              flags:
// CHECK-SAME:             ){{$}}
// CHECK: ![[ULONG]] = !MDBasicType(name: "long unsigned int"
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "C"
// CHECK-SAME:             line: 5
// CHECK-NOT:              baseType:
// CHECK-SAME:             size: 32, align: 32
// CHECK-NOT:              offset:
// CHECK-NOT:              flags:
// CHECK-SAME:             ){{$}}

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

namespace test2 {
// FIXME: this should just be a declaration under -fno-standalone-debug
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST2:![0-9]+]]
// CHECK-SAME:             elements: [[TEST_ENUMS:![0-9]+]]
// CHECK-SAME:             identifier: "_ZTSN5test21EE"
// CHECK: [[TEST2]] = !MDNamespace(name: "test2"
// CHECK: [[TEST_ENUMS]] = !{[[TEST_E:![0-9]*]]}
// CHECK: [[TEST_E]] = !MDEnumerator(name: "e", value: 0)
enum E : int;
void func(E *) {
}
enum E : int { e };
}

namespace test3 {
// FIXME: this should just be a declaration under -fno-standalone-debug
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST3:![0-9]+]]
// CHECK-SAME:             elements: [[TEST_ENUMS]]
// CHECK-SAME:             identifier: "_ZTSN5test31EE"
// CHECK: [[TEST3]] = !MDNamespace(name: "test3"
enum E : int { e };
void func(E *) {
}
}

namespace test4 {
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST4:![0-9]+]]
// CHECK-SAME:             elements: [[TEST_ENUMS]]
// CHECK-SAME:             identifier: "_ZTSN5test41EE"
// CHECK: [[TEST4]] = !MDNamespace(name: "test4"
enum E : int;
void f1(E *) {
}
enum E : int { e };
void f2(E) {
}
}

// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "D"
// CHECK-SAME:             line: 6
// CHECK-SAME:             size: 16, align: 16
// CHECK-NOT:              offset:
// CHECK-SAME:             flags: DIFlagFwdDecl

namespace test5 {
// CHECK: !MDCompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST5:![0-9]+]]
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTSN5test51EE"
// CHECK: [[TEST5]] = !MDNamespace(name: "test5"
enum E : int;
void f1(E *) {
}
}

namespace test6 {
// Ensure typedef'd enums aren't manifest by debug info generation.
// This could cause "typedef changes linkage of anonymous type, but linkage was
// already computed" errors.
// CHECK-NOT: test6
typedef enum {
} E;
}
