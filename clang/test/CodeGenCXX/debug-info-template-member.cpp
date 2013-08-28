// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

struct MyClass {
  template <int i> int add(int j) {
    return i + j;
  }
  virtual void func() {
  }
};

int add2(int x) {
  return MyClass().add<2>(x);
}

inline int add3(int x) {
  return MyClass().add<3>(x); // even though add<3> is ODR used, don't emit it since we don't codegen it
}

// CHECK: [[C:![0-9]*]] = {{.*}}, metadata [[C_MEM:![0-9]*]], i32 0, metadata [[C]], null, null} ; [ DW_TAG_structure_type ] [MyClass]
// CHECK: [[C_MEM]] = metadata !{metadata [[C_VPTR:![0-9]*]], metadata [[C_ADD:![0-9]*]], metadata [[C_FUNC:![0-9]*]], metadata [[C_CTOR:![0-9]*]]}
// CHECK: [[C_VPTR]] = {{.*}} ; [ DW_TAG_member ] [_vptr$MyClass]
// CHECK: [[C_ADD]] = {{.*}} ; [ DW_TAG_subprogram ] [line 4] [add<2>]
// CHECK: [[C_FUNC]] = {{.*}} ; [ DW_TAG_subprogram ] [line 7] [func]
// CHECK: [[C_CTOR]] = {{.*}} ; [ DW_TAG_subprogram ] [line 0] [MyClass]

template<typename T>
struct outer {
  struct inner {
    int i;
  };
};

struct foo {
  void func(outer<foo>::inner);
};

inline void func() {
  // require 'foo' to be complete before the emission of 'inner' so that, when
  // constructing the context chain for 'x' we emit the full definition of
  // 'foo', which requires the definition of 'inner' again
  foo f;
}

outer<foo>::inner x;

// CHECK: metadata [[OUTER_FOO_INNER:![0-9]*]], i32 {{[0-9]*}}, i32 {{[0-9]*}}, %"struct.outer<foo>::inner"* @x, {{.*}} ; [ DW_TAG_variable ] [x]
// CHECK: [[OUTER_FOO_INNER]] = {{.*}} ; [ DW_TAG_structure_type ] [inner]
// CHECK: [[FOO_MEM:![0-9]*]], i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo]
// CHECK: [[FOO_MEM]] = metadata !{metadata [[FOO_FUNC:![0-9]*]]}
// CHECK: [[FOO_FUNC]] = {{.*}}, metadata !"_ZN3foo4funcEN5outerIS_E5innerE", i32 {{[0-9]*}}, metadata [[FOO_FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [func]
// CHECK: [[FOO_FUNC_TYPE]] = {{.*}}, metadata [[FOO_FUNC_PARAMS:![0-9]*]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FOO_FUNC_PARAMS]] = metadata !{null, metadata !{{[0-9]*}}, metadata [[OUTER_FOO_INNER]]}
