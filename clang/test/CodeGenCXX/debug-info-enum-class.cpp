// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
enum D : short; // enum forward declaration
A a;
B b;
C c;
D d;

// CHECK: ; [ DW_TAG_enumeration_type ] [A] [line 3, size 32, align 32, offset 0] [def] [from int]
// CHECK: ; [ DW_TAG_enumeration_type ] [B] [line 4, size 64, align 64, offset 0] [def] [from long unsigned int]
// CHECK: ; [ DW_TAG_enumeration_type ] [C] [line 5, size 32, align 32, offset 0] [def] [from ]

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
// CHECK: metadata !{i32 {{[^,]*}}, {{[^,]*}}, metadata [[TEST2:![0-9]*]], {{.*}}, metadata [[TEST_ENUMS:![0-9]*]], {{[^,]*}}, null, null, metadata !"_ZTSN5test21EE"} ; [ DW_TAG_enumeration_type ] [E]
// CHECK: [[TEST2]] = {{.*}} ; [ DW_TAG_namespace ] [test2]
// CHECK: [[TEST_ENUMS]] = metadata !{metadata [[TEST_E:![0-9]*]]}
// CHECK: [[TEST_E]] = {{.*}}, metadata !"e", i64 0} ; [ DW_TAG_enumerator ] [e :: 0]
enum E : int;
void func(E *) {
}
enum E : int { e };
}

namespace test3 {
// FIXME: this should just be a declaration under -fno-standalone-debug
// CHECK: metadata !{i32 {{[^,]*}}, {{[^,]*}}, metadata [[TEST3:![0-9]*]], {{.*}}, metadata [[TEST_ENUMS]], {{[^,]*}}, null, null, metadata !"_ZTSN5test31EE"} ; [ DW_TAG_enumeration_type ] [E]
// CHECK: [[TEST3]] = {{.*}} ; [ DW_TAG_namespace ] [test3]
enum E : int { e };
void func(E *) {
}
}

namespace test4 {
// CHECK: metadata !{i32 {{[^,]*}}, {{[^,]*}}, metadata [[TEST4:![0-9]*]], {{.*}}, metadata [[TEST_ENUMS]], {{[^,]*}}, null, null, metadata !"_ZTSN5test41EE"} ; [ DW_TAG_enumeration_type ] [E]
// CHECK: [[TEST4]] = {{.*}} ; [ DW_TAG_namespace ] [test4]
enum E : int;
void f1(E *) {
}
enum E : int { e };
void f2(E) {
}
}

// CHECK: ; [ DW_TAG_enumeration_type ] [D] [line 6, size 16, align 16, offset 0] [decl] [from ]

namespace test5 {
// CHECK: metadata !{i32 {{[^,]*}}, {{[^,]*}}, metadata [[TEST5:![0-9]*]], {{.*}}, null, {{[^,]*}}, null, null, metadata !"_ZTSN5test51EE"} ; [ DW_TAG_enumeration_type ] [E]
// CHECK: [[TEST5]] = {{.*}} ; [ DW_TAG_namespace ] [test5]
enum E : int;
void f1(E *) {
}
}

namespace test6 {
// Ensure typedef'd enums aren't manifest by debug info generation.
// This could cause "typedef changes linkage of anonymous type, but linkage was
// already computed" errors.
// CHECK-NOT: test7
typedef enum {
} E;
}
