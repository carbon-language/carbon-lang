// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++11 [basic.link]p6:
//   The name of a function declared in block scope and the name
//   of a variable declared by a block scope extern declaration
//   have linkage. If there is a visible declaration of an entity
//   with linkage having the same name and type, ignoring entities
//   declared outside the innermost enclosing namespace scope, the
//   block scope declaration declares that same entity and
//   receives the linkage of the previous declaration.

// rdar://13535367
namespace test0 {
  extern "C" int test0_array[];
  void declare() { extern int test0_array[100]; }
  extern "C" int test0_array[];
  int value = sizeof(test0_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace test1 {
  extern "C" int test1_array[];
  void test() {
    { extern int test1_array[100]; }
    extern int test1_array[];
    int x = sizeof(test1_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  }
}

namespace test2 {
  void declare() { extern int test2_array[100]; }
  extern int test2_array[];
  int value = sizeof(test2_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

namespace test3 {
  void test() {
    { extern int test3_array[100]; }
    extern int test3_array[];
    int x = sizeof(test3_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
  }
}


