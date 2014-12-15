// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only -std=c++14 -emit-llvm -g %s -o - | FileCheck %s

// CHECK: [[EMPTY:![0-9]*]] = !{}
// CHECK: \00foo\00{{.*}}, [[EMPTY]], {{.*}}} ; [ DW_TAG_structure_type ]
// FIXME: The context of this definition should be the CU/file scope, not the class.
// CHECK: !"_ZTS3foo", [[SUBROUTINE_TYPE:![0-9]*]], {{.*}}, [[FUNC_DECL:![0-9]*]], {{![0-9]*}}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[SUBROUTINE_TYPE]] = {{.*}}, [[TYPE_LIST:![0-9]*]],
// CHECK: [[TYPE_LIST]] = !{[[INT:![0-9]*]]}
// CHECK: [[INT]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: [[FUNC_DECL]] = {{.*}}, !"_ZTS3foo", [[SUBROUTINE_TYPE]], {{.*}}} ; [ DW_TAG_subprogram ] {{.*}} [func]

struct foo {
  static auto func();
};

foo f;

auto foo::func() {
  return 1;
}
