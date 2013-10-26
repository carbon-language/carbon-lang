// RUN: %clang -g -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -g -gline-tables-only -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-GMLT %s
// RUN: %clang -g -fno-limit-debug-info -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-NOLIMIT %s

namespace A {
#line 1 "foo.cpp"
namespace B {
int i;
void f1() { }
void f1(int) { }
struct foo;
struct bar { };
typedef bar baz;
}
}
namespace A {
using namespace B;
}

using namespace A;
namespace E = A;

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
  using B::baz;
  namespace X = A;
  namespace Y = X;
  return i + X::B::i + Y::B::i;
}

// This should work even if 'i' and 'func' were declarations & not definitions,
// but it doesn't yet.

// CHECK: [[CU:![0-9]*]] = {{.*}}[[MODULES:![0-9]*]], metadata !""} ; [ DW_TAG_compile_unit ]
// CHECK: [[FILE:![0-9]*]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[FOO:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
// CHECK: [[FOOCPP:![0-9]*]] = metadata !{metadata !"foo.cpp", {{.*}}
// CHECK: [[NS:![0-9]*]] = {{.*}}, metadata [[FILE2:![0-9]*]], metadata [[CTXT:![0-9]*]], {{.*}} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = {{.*}}, metadata [[FILE]], null, {{.*}} ; [ DW_TAG_namespace ] [A] [line 5]
// CHECK: [[BAR:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [decl] [from ]
// CHECK: [[F1:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
// CHECK: [[FUNC:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[FILE2]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
// CHECK: [[I:![0-9]*]] = {{.*}}, metadata [[NS]], metadata !"i", {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[MODULES]] = metadata !{metadata [[M1:![0-9]*]], metadata [[M2:![0-9]*]], metadata [[M3:![0-9]*]], metadata [[M4:![0-9]*]], metadata [[M5:![0-9]*]], metadata [[M6:![0-9]*]], metadata [[M7:![0-9]*]], metadata [[M8:![0-9]*]], metadata [[M9:![0-9]*]], metadata [[M10:![0-9]*]], metadata [[M11:![0-9]*]], metadata [[M12:![0-9]*]]}
// CHECK: [[M1]] = metadata !{i32 {{[0-9]*}}, metadata [[CTXT]], metadata [[NS]], i32 11} ; [ DW_TAG_imported_module ]
// CHECK: [[M2]] = metadata !{i32 {{[0-9]*}}, metadata [[CU]], metadata [[CTXT]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_module ]
// CHECK: [[M3]] = metadata !{i32 {{[0-9]*}}, metadata [[CU]], metadata [[CTXT]], i32 15, metadata !"E"} ; [ DW_TAG_imported_module ]
// CHECK: [[M4]] = metadata !{i32 {{[0-9]*}}, metadata [[LEX2:![0-9]*]], metadata [[NS]], i32 19} ; [ DW_TAG_imported_module ]
// CHECK: [[LEX2]] = metadata !{i32 {{[0-9]*}}, metadata [[FILE2]], metadata [[LEX1:![0-9]+]], i32 {{[0-9]*}}, i32 0, i32 {{.*}}} ; [ DW_TAG_lexical_block ]
// CHECK: [[LEX1]] = metadata !{i32 {{[0-9]*}}, metadata [[FILE2]], metadata [[FUNC]], i32 {{[0-9]*}}, i32 0, i32 {{.*}}} ; [ DW_TAG_lexical_block ]
// CHECK: [[M5]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[CTXT]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_module ]
// CHECK: [[M6]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[FOO:![0-9]*]], i32 23} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M7]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[BAR:![0-9]*]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M8]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[F1]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M9]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[I]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M10]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[BAZ:![0-9]*]], i32 {{[0-9]*}}} ; [ DW_TAG_imported_declaration ]
// CHECK: [[BAZ]] = metadata !{i32 {{[0-9]*}}, metadata [[FOOCPP]], metadata [[NS]], {{.*}}, metadata !"_ZTSN1A1B3barE"} ; [ DW_TAG_typedef ] [baz] {{.*}} [from _ZTSN1A1B3barE]
// CHECK: [[M11]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[CTXT]], i32 {{[0-9]*}}, metadata !"X"} ; [ DW_TAG_imported_module ]
// CHECK: [[M12]] = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], metadata [[M11]], i32 {{[0-9]*}}, metadata !"Y"} ; [ DW_TAG_imported_module ]

// CHECK-GMLT: [[CU:![0-9]*]] = {{.*}}[[MODULES:![0-9]*]], metadata !""} ; [ DW_TAG_compile_unit ]
// CHECK-GMLT: [[MODULES]] = metadata !{i32 0}

// CHECK-NOLIMIT: ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [def] [from ]

// FIXME: It is confused on win32 to generate file entry when dosish filename is given.
// REQUIRES: shell
// REQUIRES: shell-preserves-root
