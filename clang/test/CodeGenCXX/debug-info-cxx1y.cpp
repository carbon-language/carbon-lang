// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only -std=c++14 -emit-llvm -g %s -o - | FileCheck %s

// CHECK: [[EMPTY:![0-9]*]] = metadata !{}
// CHECK: \00foo\00{{.*}}, metadata [[EMPTY]], {{.*}}} ; [ DW_TAG_structure_type ]
// FIXME: The context of this definition should be the CU/file scope, not the class.
// CHECK: metadata !"_ZTS3foo", metadata [[SUBROUTINE_TYPE:![0-9]*]], {{.*}}, metadata [[FUNC_DECL:![0-9]*]], metadata {{![0-9]*}}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[SUBROUTINE_TYPE]] = {{.*}}, metadata [[TYPE_LIST:![0-9]*]],
// CHECK: [[TYPE_LIST]] = metadata !{metadata [[INT:![0-9]*]]}
// CHECK: [[INT]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: [[FUNC_DECL]] = {{.*}}, metadata !"_ZTS3foo", metadata [[SUBROUTINE_TYPE]], {{.*}}} ; [ DW_TAG_subprogram ] {{.*}} [func]

struct foo {
  static auto func();
};

foo f;

auto foo::func() {
  return 1;
}
