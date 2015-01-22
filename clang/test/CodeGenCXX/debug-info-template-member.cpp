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

// CHECK: [[FOO_MEM:![0-9]*]], null, null, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: [[FOO_MEM]] = !{[[FOO_FUNC:![0-9]*]]}
// CHECK: [[FOO_FUNC]] = !{!"0x2e\00func\00func\00_ZN3foo4funcEN5outerIS_E5innerE\00{{.*}}"{{, [^,]+, [^,]+}}, [[FOO_FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [func]
// CHECK: [[FOO_FUNC_TYPE]] = {{.*}}, [[FOO_FUNC_PARAMS:![0-9]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FOO_FUNC_PARAMS]] = !{null, !{{[0-9]*}}, !"[[OUTER_FOO_INNER_ID:.*]]"}
// CHECK: !{{[0-9]*}} = {{.*}}, null, !"[[OUTER_FOO_INNER_ID]]"} ; [ DW_TAG_structure_type ] [inner]

// CHECK: [[VIRT_MEM:![0-9]*]], !"_ZTS4virtI4elemE", [[VIRT_TEMP_PARAM:![0-9]*]], !"_ZTS4virtI4elemE"} ; [ DW_TAG_structure_type ] [virt<elem>] {{.*}} [def]
// CHECK: [[VIRT_TEMP_PARAM]] = !{[[VIRT_T:![0-9]*]]}
// CHECK: [[VIRT_T]] = !{!"0x2f\00T\000\000"{{, [^,]+}}, !"_ZTS4elem", {{.*}} ; [ DW_TAG_template_type_parameter ]

// CHECK: [[C:![0-9]*]] = {{.*}}, [[C_MEM:![0-9]*]], !"_ZTS7MyClass", null, !"_ZTS7MyClass"} ; [ DW_TAG_structure_type ] [MyClass]
// CHECK: [[C_MEM]] = !{[[C_VPTR:![0-9]*]], [[C_FUNC:![0-9]*]]}
// CHECK: [[C_VPTR]] = {{.*}} ; [ DW_TAG_member ] [_vptr$MyClass]

// CHECK: [[C_FUNC]] = {{.*}} ; [ DW_TAG_subprogram ] [line 7] [func]

// CHECK: [[ELEM:![0-9]*]] = {{.*}}, [[ELEM_MEM:![0-9]*]], null, null, !"_ZTS4elem"} ; [ DW_TAG_structure_type ] [elem] {{.*}} [def]
// CHECK: [[ELEM_MEM]] = !{[[ELEM_X:![0-9]*]]}
// CHECK: [[ELEM_X]] = {{.*}} ; [ DW_TAG_member ] [x] {{.*}} [static] [from _ZTS4virtI4elemE]

// Check that the member function template specialization and implicit special
// members (the default ctor) refer to their class by scope, even though they
// didn't appear in the class's member list (C_MEM). This prevents the functions
// from being added to type units, while still appearing in the type
// declaration/reference in the compile unit.
// CHECK: !"_ZTS7MyClass", {{.*}} ; [ DW_TAG_subprogram ] [line 0] [MyClass]
// CHECK: !"_ZTS7MyClass", {{.*}} ; [ DW_TAG_subprogram ] [line 4] [add<2>]

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

// CHECK:  !"0x34\00{{.*}}", {{.*}}, !"[[OUTER_FOO_INNER_ID]]", %"struct.outer<foo>::inner"* @x, {{.*}} ; [ DW_TAG_variable ] [x]

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

