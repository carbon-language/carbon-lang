// RUN: %clang_cc1 -g -fno-standalone-debug -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -g -gline-tables-only    -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-GMLT %s
// RUN: %clang_cc1 -g -fstandalone-debug    -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-NOLIMIT %s

namespace A {
#line 1 "foo.cpp"
namespace B {
extern int i;
int f1() { return 0; }
void f1(int) { }
struct foo;
struct bar { };
typedef bar baz;
extern int var_decl;
void func_decl(void);
extern int var_fwd;
void func_fwd(void);
}
}
namespace A {
using namespace B;
}

using namespace A;
namespace E = A;
int B::i = f1();
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
  using B::var_decl;
  using B::func_decl;
  using B::var_fwd;
  using B::func_fwd;
  return i + X::B::i + Y::B::i;
}

namespace A {
using B::i;
namespace B {
int var_fwd = i;
}
}
void B::func_fwd() {}

// This should work even if 'i' and 'func' were declarations & not definitions,
// but it doesn't yet.

// CHECK: [[CU:![0-9]*]] = !{!"0x11\00{{.*}}\001"{{.*}}, [[MODULES:![0-9]*]]} ; [ DW_TAG_compile_unit ]
// CHECK: [[FOO:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
// CHECK: [[FOOCPP:![0-9]*]] = !{!"foo.cpp", {{.*}}
// CHECK: [[NS:![0-9]*]] = !{!"0x39\00B\001", [[FILE2:![0-9]*]], [[CTXT:![0-9]*]]} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = !{!"0x39\00A\005", [[FILE:![0-9]*]], null} ; [ DW_TAG_namespace ] [A] [line 5]
// CHECK: [[FILE]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[BAR:![0-9]*]] {{.*}} ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [decl] [from ]
// CHECK: [[F1:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
// CHECK: [[FILE2]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
// CHECK: [[FUNC:![0-9]*]] {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[FUNC_FWD:![0-9]*]] {{.*}} [ DW_TAG_subprogram ] [line 47] [def] [func_fwd]
// CHECK: [[I:![0-9]*]] = !{!"0x34\00i\00{{.*}}", [[NS]], {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[VAR_FWD:![0-9]*]] = !{!"0x34\00var_fwd\00{{.*}}", [[NS]], {{.*}}} ; [ DW_TAG_variable ] [var_fwd] [line 44] [def]

// CHECK: [[MODULES]] = !{[[M1:![0-9]*]], [[M2:![0-9]*]], [[M3:![0-9]*]], [[M4:![0-9]*]], [[M5:![0-9]*]], [[M6:![0-9]*]], [[M7:![0-9]*]], [[M8:![0-9]*]], [[M9:![0-9]*]], [[M10:![0-9]*]], [[M11:![0-9]*]], [[M12:![0-9]*]], [[M13:![0-9]*]], [[M14:![0-9]*]], [[M15:![0-9]*]], [[M16:![0-9]*]], [[M17:![0-9]*]]}
// CHECK: [[M1]] = !{!"0x3a\0015\00", [[CTXT]], [[NS]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M2]] = !{!"0x3a\00{{[0-9]+}}\00", [[CU]], [[CTXT]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M3]] = !{!"0x8\0019\00E", [[CU]], [[CTXT]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M4]] = !{!"0x3a\0023\00", [[LEX2:![0-9]*]], [[NS]]} ; [ DW_TAG_imported_module ]
// CHECK: [[LEX2]] = !{!"0xb\00{{[0-9]*}}\000\00{{.*}}", [[FILE2]], [[LEX1:![0-9]+]]} ; [ DW_TAG_lexical_block ]
// CHECK: [[LEX1]] = !{!"0xb\00{{[0-9]*}}\000\00{{.*}}", [[FILE2]], [[FUNC]]} ; [ DW_TAG_lexical_block ]
// CHECK: [[M5]] = !{!"0x3a\00{{[0-9]+}}\00", [[FUNC]], [[CTXT]]} ; [ DW_TAG_imported_module ]
// CHECK: [[M6]] = !{!"0x8\0027\00", [[FUNC]], [[FOO:!"_ZTSN1A1B3fooE"]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M7]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[BAR:!"_ZTSN1A1B3barE"]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M8]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[F1]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M9]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[I]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M10]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[BAZ:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[BAZ]] = !{!"0x16\00baz\00{{.*}}", [[FOOCPP]], [[NS]], !"_ZTSN1A1B3barE"} ; [ DW_TAG_typedef ] [baz] {{.*}} [from _ZTSN1A1B3barE]
// CHECK: [[M11]] = !{!"0x8\00{{[0-9]+}}\00X", [[FUNC]], [[CTXT]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M12]] = !{!"0x8\00{{[0-9]+}}\00Y", [[FUNC]], [[M11]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M13]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[VAR_DECL:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK [[VAR_DECL]] = !{!"0x34\00var_decl\00{{.*}}", [[NS]], {{.*}}} ; [ DW_TAG_variable ] [var_decl] [line 8]
// CHECK: [[M14]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[FUNC_DECL:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[FUNC_DECL]] = !{!"0x2e\00func_decl\00{{.*}}", [[FOOCPP]], [[NS]], {{.*}}} ; [ DW_TAG_subprogram ] [line 9] [scope 0] [func_decl]
// CHECK: [[M15]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[VAR_FWD:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M16]] = !{!"0x8\00{{[0-9]+}}\00", [[FUNC]], [[FUNC_FWD:![0-9]*]]} ; [ DW_TAG_imported_declaration ]
// CHECK: [[M17]] = !{!"0x8\00{{[0-9]+}}\00", [[CTXT]], [[I]]} ; [ DW_TAG_imported_declaration ]
// CHECK-GMLT: [[CU:![0-9]*]] = !{!"0x11\00{{.*}}\002"{{.*}}, [[MODULES:![0-9]*]]} ; [ DW_TAG_compile_unit ]
// CHECK-GMLT: [[MODULES]] = !{}

// CHECK-NOLIMIT: ; [ DW_TAG_structure_type ] [bar] [line 6, {{.*}}] [def] [from ]

// REQUIRES: shell-preserves-root
// REQUIRES: dw2
