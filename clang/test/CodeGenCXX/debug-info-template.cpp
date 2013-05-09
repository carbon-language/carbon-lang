// RUN: %clang_cc1 -emit-llvm -g %s -o - -std=c++11 | FileCheck %s

// CHECK: [[INT:![0-9]*]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: metadata [[TCI:![0-9]*]], i32 0, i32 1, %class.TC* @tci, null} ; [ DW_TAG_variable ] [tci]
// CHECK: [[TC:![0-9]*]] = {{.*}}, metadata [[TCARGS:![0-9]*]]} ; [ DW_TAG_class_type ] [TC<int, 2, &glb, &foo::e, &foo::f, nullptr>]
// CHECK: [[TCARGS]] = metadata !{metadata [[TCARG1:![0-9]*]], metadata [[TCARG2:![0-9]*]]}
//
// We seem to be missing file/line/col info on template value parameters -
// metadata supports it but it's not populated.
//
// CHECK: [[TCARG1]] = {{.*}}metadata !"T", metadata [[INT]], {{.*}} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[TCARG2]] = {{.*}}metadata !"", metadata [[UINT:![0-9]*]], i64 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[UINT]] = {{.*}} ; [ DW_TAG_base_type ] [unsigned int]

struct foo {
  int e;
  void f();
};

template<typename T, unsigned, int *x, int foo::*a, void (foo::*b)(), int *n>
class TC {
};

int glb;

TC<int, 2, &glb, &foo::e, &foo::f, nullptr> tci;
