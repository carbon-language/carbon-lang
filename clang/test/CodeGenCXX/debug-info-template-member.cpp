// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

struct MyClass {
  template <int i> int add(int j) {
    return i + j;
  }
};

int add2(int x) {
  return MyClass().add<2>(x);
}

inline int add3(int x) {
  return MyClass().add<3>(x); // even though add<3> is ODR used, don't emit it since we don't codegen it
}

// CHECK: metadata [[C_MEM:![0-9]*]], i32 0, null, null, null} ; [ DW_TAG_structure_type ] [MyClass]
// CHECK: [[C_MEM]] = metadata !{metadata [[C_TEMP:![0-9]*]]}
// CHECK: [[C_TEMP]] = {{.*}} ; [ DW_TAG_subprogram ] [line 4] [add<2>]

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
// CHECK: [[FOO_FUNC_TYPE]] = {{.*}}, metadata [[FOO_FUNC_PARAMS:![0-9]*]], i32 0, null{{.*}}} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FOO_FUNC_PARAMS]] = metadata !{null, metadata !{{[0-9]*}}, metadata [[OUTER_FOO_INNER]]}
