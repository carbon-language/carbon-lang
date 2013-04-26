// RUN: %clang  -g -S -emit-llvm %s -o - | FileCheck %s

namespace A {
#line 1 "foo.cpp"
namespace B {
int i;
}
using namespace B;
}

using namespace A;

int func() {
  using namespace A::B;
  return i;
}

// CHECK: [[CU:![0-9]*]] = {{.*}}[[MODULES:![0-9]*]], metadata !""} ; [ DW_TAG_compile_unit ]
// CHECK: [[FILE:![0-9]*]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[FUNC:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 9] [def] [func]
// CHECK: [[FILE2:![0-9]*]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
// CHECK: [[VAR:![0-9]*]] = {{.*}}, metadata [[NS:![0-9]*]], metadata !"i", {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[NS]] = {{.*}}, metadata [[FILE2]], metadata [[CTXT:![0-9]*]], {{.*}} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = {{.*}}, metadata [[FILE]], null, {{.*}} ; [ DW_TAG_namespace ] [A] [line 3]
// CHECK: [[MODULES]] = metadata !{metadata [[M1:![0-9]*]], metadata [[M2:![0-9]*]], metadata [[M3:![0-9]*]]}
// CHECK: [[M1]] = metadata !{i32 {{[0-9]*}}, metadata [[CTXT]], metadata [[NS]], i32 4} ; [ DW_TAG_imported_module ]
// CHECK: [[M2]] = metadata !{i32 {{[0-9]*}}, metadata [[CU]], metadata [[CTXT]], i32 7} ; [ DW_TAG_imported_module ]
// CHECK: [[M3]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[NS]], i32 10} ; [ DW_TAG_imported_module ]

// FIXME: It is confused on win32 to generate file entry when dosish filename is given.
// REQUIRES: shell
