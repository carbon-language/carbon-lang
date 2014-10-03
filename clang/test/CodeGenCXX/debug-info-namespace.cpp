// RUN: %clang_cc1 -g -fno-standalone-debug -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -g -gline-tables-only    -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-GMLT %s
// RUN: %clang_cc1 -g -fstandalone-debug    -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-NOLIMIT %s

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

namespace A {
using B::i;
}

// This should work even if 'i' and 'func' were declarations & not definitions,
// but it doesn't yet.

// CHECK: [[CU:![0-9]*]] = metadata !{metadata !"0x11\00{{.*}}\001"{{.*}}, metadata [[MODULES:![0-9]*]]} ; [ DW_TAG_compile_unit ]
// CHECK: [[FOO:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
// CHECK: [[FOOCPP:![0-9]*]] = metadata !{metadata !"foo.cpp", {{.*}}
// CHECK: [[NS:![0-9]*]] = metadata !{metadata !"0x39\00B\001", metadata [[FILE2:![0-9]*]], metadata [[CTXT:![0-9]*]]} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = metadata !{metadata !"0x39\00A\005", metadata [[FILE:![0-9]*]], null} ; [ DW_TAG_namespace ] [A] [line 5]
// CHECK: [[FILE]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[BAR:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [decl] [from ]
// CHECK: [[F1:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
// CHECK: [[FUNC:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[FILE2]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
// CHECK: [[I:![0-9]*]] = metadata !{metadata !"0x34\00i\00{{.*}}", metadata [[NS]], {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[MODULES]] = metadata !{metadata [[M1:![0-9]*]], metadata [[M2:![0-9]*]], metadata [[M3:![0-9]*]], metadata [[M4:![0-9]*]], metadata [[M5:![0-9]*]], metadata [[M6:![0-9]*]], metadata [[M7:![0-9]*]], metadata [[M8:![0-9]*]], metadata [[M9:![0-9]*]], metadata [[M10:![0-9]*]], metadata [[M11:![0-9]*]], metadata [[M12:![0-9]*]], metadata [[M13:![0-9]*]]}
// CHECK: [[M1]] = metadata !{metadata !"0x3a\0011\00", metadata [[CTXT]], metadata [[NS]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M2]] = metadata !{metadata !"0x3a\00{{[0-9]+}}\00", metadata [[CU]], metadata [[CTXT]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M3]] = metadata !{metadata !"0x8\0015\00E", metadata [[CU]], metadata [[CTXT]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M4]] = metadata !{metadata !"0x3a\0019\00", metadata [[LEX2:![0-9]*]], metadata [[NS]]} ; [ DW_TAG_imported_module ]
// CHECK: [[LEX2]] = metadata !{metadata !"0xb\00{{[0-9]*}}\000\00{{.*}}", metadata [[FILE2]], metadata [[LEX1:![0-9]+]]} ; [ DW_TAG_lexical_block ]
// CHECK: [[LEX1]] = metadata !{metadata !"0xb\00{{[0-9]*}}\000\00{{.*}}", metadata [[FILE2]], metadata [[FUNC]]} ; [ DW_TAG_lexical_block ]
// CHECK: [[M5]] = metadata !{metadata !"0x3a\00{{[0-9]+}}\00", metadata [[FUNC]], metadata [[CTXT]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M6]] = metadata !{metadata !"0x8\0023\00", metadata [[FUNC]], metadata [[FOO:!"_ZTSN1A1B3fooE"]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M7]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00", metadata [[FUNC]], metadata [[BAR:!"_ZTSN1A1B3barE"]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M8]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00", metadata [[FUNC]], metadata [[F1]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M9]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00", metadata [[FUNC]], metadata [[I]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M10]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00", metadata [[FUNC]], metadata [[BAZ:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[BAZ]] = metadata !{metadata !"0x16\00baz\00{{.*}}", metadata [[FOOCPP]], metadata [[NS]], metadata !"_ZTSN1A1B3barE"} ; [ DW_TAG_typedef ] [baz] {{.*}} [from _ZTSN1A1B3barE]
// CHECK: [[M11]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00X", metadata [[FUNC]], metadata [[CTXT]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M12]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00Y", metadata [[FUNC]], metadata [[M11]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M13]] = metadata !{metadata !"0x8\00{{[0-9]+}}\00", metadata [[CTXT]], metadata [[I]]} ; [ DW_TAG_imported_declaration ]

// CHECK-GMLT: [[CU:![0-9]*]] = metadata !{metadata !"0x11\00{{.*}}\002"{{.*}}, metadata [[MODULES:![0-9]*]]} ; [ DW_TAG_compile_unit ]
// CHECK-GMLT: [[MODULES]] = metadata !{}

// CHECK-NOLIMIT: ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [def] [from ]

// FIXME: It is confused on win32 to generate file entry when dosish filename is given.
// REQUIRES: shell
// REQUIRES: shell-preserves-root
// REQUIRES: dw2
