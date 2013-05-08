// RUN: %clang  -g -S -emit-llvm %s -o - | FileCheck %s

namespace A {
#line 1 "foo.cpp"
namespace B {
int i;
void f1() { }
void f1(int) { }
struct foo;
struct bar { };
}
using namespace B;
}

using namespace A;

int func(bool b) {
  if (b) {
    using namespace A::B;
    return i;
  }
  using namespace A;
  using B::foo;
  using B::bar;
  using B::f1;
  using B::i;
  bar x;
  return i;
}

// This should work even if 'i' and 'func' were declarations & not definitions,
// but it doesn't yet.

// CHECK: [[CU:![0-9]*]] = {{.*}}[[MODULES:![0-9]*]], metadata !""} ; [ DW_TAG_compile_unit ]
// CHECK: [[FILE:![0-9]*]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[NS:![0-9]*]] = {{.*}}, metadata [[FILE2:![0-9]*]], metadata [[CTXT:![0-9]*]], {{.*}} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = {{.*}}, metadata [[FILE]], null, {{.*}} ; [ DW_TAG_namespace ] [A] [line 3]
// CHECK: [[F1:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
// CHECK: [[FUNC:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 13] [def] [func]
// CHECK: [[FILE2]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
// CHECK: [[I:![0-9]*]] = {{.*}}, metadata [[NS]], metadata !"i", {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[MODULES]] = metadata !{metadata [[M1:![0-9]*]], metadata [[M2:![0-9]*]], metadata [[M3:![0-9]*]], metadata [[M4:![0-9]*]], metadata [[M5:![0-9]*]], metadata [[M6:![0-9]*]], metadata [[M7:![0-9]*]], metadata [[M8:![0-9]*]]}
// CHECK: [[M1]] = metadata !{i32 {{[0-9]*}}, metadata [[CTXT]], metadata [[NS]], i32 8} ; [ DW_TAG_imported_module ]
// CHECK: [[M2]] = metadata !{i32 {{[0-9]*}}, metadata [[CU]], metadata [[CTXT]], i32 11} ; [ DW_TAG_imported_module ]
// CHECK: [[M3]] = metadata !{i32 {{[0-9]*}}, metadata [[LEX:![0-9]*]], metadata [[NS]], i32 15} ; [ DW_TAG_imported_module ]
// CHECK: [[LEX]] = metadata !{i32 {{[0-9]*}}, metadata [[FILE2]], metadata [[FUNC]], i32 14, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
// CHECK: [[M4]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[CTXT]], i32 18} ; [ DW_TAG_imported_module ]
// CHECK: [[M5]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[FOO:![0-9]*]], i32 19} ; [ DW_TAG_imported_declaration ]
// CHECK: [[FOO]] {{.*}} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [fwd] [from ]
// CHECK: [[M6]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[BAR:![0-9]*]], i32 20} ; [ DW_TAG_imported_declaration ]
// CHECK: [[BAR]] {{.*}} ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [from ]
// CHECK: [[M7]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[F1]], i32 21} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M8]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[I]], i32 22} ; [ DW_TAG_imported_declaration ]

// FIXME: It is confused on win32 to generate file entry when dosish filename is given.
// REQUIRES: shell
