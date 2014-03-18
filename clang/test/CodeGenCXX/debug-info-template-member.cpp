// RUN: %clang_cc1 -emit-llvm -g -fno-standalone-debug -triple x86_64-apple-darwin %s -o - | FileCheck %s

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

// CHECK: [[FOO_MEM:![0-9]*]], i32 0, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: [[FOO_MEM]] = metadata !{metadata [[FOO_FUNC:![0-9]*]]}
// CHECK: [[FOO_FUNC]] = {{.*}}, metadata !"_ZN3foo4funcEN5outerIS_E5innerE", i32 {{[0-9]*}}, metadata [[FOO_FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [func]
// CHECK: [[FOO_FUNC_TYPE]] = {{.*}}, metadata [[FOO_FUNC_PARAMS:![0-9]*]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FOO_FUNC_PARAMS]] = metadata !{null, metadata !{{[0-9]*}}, metadata [[OUTER_FOO_INNER:![0-9]*]]}
// CHECK: [[OUTER_FOO_INNER]] = {{.*}}, null, metadata !"[[OUTER_FOO_INNER_ID:.*]]"} ; [ DW_TAG_structure_type ] [inner]

// CHECK: metadata [[VIRT_MEM:![0-9]*]], i32 0, metadata !"_ZTS4virtI4elemE", metadata [[VIRT_TEMP_PARAM:![0-9]*]], metadata !"_ZTS4virtI4elemE"} ; [ DW_TAG_structure_type ] [virt<elem>] {{.*}} [def]
// CHECK: [[VIRT_TEMP_PARAM]] = metadata !{metadata [[VIRT_T:![0-9]*]]}
// CHECK: [[VIRT_T]] = {{.*}}, metadata !"T", metadata !"_ZTS4elem", {{.*}} ; [ DW_TAG_template_type_parameter ]

// CHECK: [[C:![0-9]*]] = {{.*}}, metadata [[C_MEM:![0-9]*]], i32 0, metadata !"_ZTS7MyClass", null, metadata !"_ZTS7MyClass"} ; [ DW_TAG_structure_type ] [MyClass]
// CHECK: [[C_MEM]] = metadata !{metadata [[C_VPTR:![0-9]*]], metadata [[C_ADD:![0-9]*]], metadata [[C_FUNC:![0-9]*]], metadata [[C_CTOR:![0-9]*]]}
// CHECK: [[C_VPTR]] = {{.*}} ; [ DW_TAG_member ] [_vptr$MyClass]

// CHECK: [[C_ADD]] = {{.*}} ; [ DW_TAG_subprogram ] [line 4] [add<2>]
// CHECK: [[C_FUNC]] = {{.*}} ; [ DW_TAG_subprogram ] [line 7] [func]
// CHECK: [[C_CTOR]] = {{.*}} ; [ DW_TAG_subprogram ] [line 0] [MyClass]

// CHECK: [[ELEM:![0-9]*]] = {{.*}}, metadata [[ELEM_MEM:![0-9]*]], i32 0, null, null, metadata !"_ZTS4elem"} ; [ DW_TAG_structure_type ] [elem] {{.*}} [def]
// CHECK: [[ELEM_MEM]] = metadata !{metadata [[ELEM_X:![0-9]*]]}
// CHECK: [[ELEM_X]] = {{.*}} ; [ DW_TAG_member ] [x] {{.*}} [static] [from _ZTS4virtI4elemE]

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

// CHECK: metadata !"[[OUTER_FOO_INNER_ID]]", i32 {{[0-9]*}}, i32 {{[0-9]*}}, %"struct.outer<foo>::inner"* @x, {{.*}} ; [ DW_TAG_variable ] [x]

template <typename T>
struct virt {
  T* values;
  virtual ~virt();
};
struct elem {
  static virt<elem> x; // ensure that completing 'elem' will require/completing 'virt<elem>'
};
inline void f1() {
  elem e; // ensure 'elem' is required to be complete when it is emitted as a template argument for 'virt<elem>'
};
void f2() {
  virt<elem> d; // emit 'virt<elem>'
}

