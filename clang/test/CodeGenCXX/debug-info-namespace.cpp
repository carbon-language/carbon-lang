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

// CHECK: [[CU:![0-9]+]] = !MDCompileUnit(
// CHECK-SAME:                            imports: [[MODULES:![0-9]*]]
// CHECK: [[FOO:![0-9]+]] = !MDCompositeType(tag: DW_TAG_structure_type, name: "foo",
// CHECK-SAME:                               line: 5
// CHECK-SAME:                               DIFlagFwdDecl
// CHECK: [[FOOCPP:![0-9]+]] = !MDFile(filename: "foo.cpp"
// CHECK: [[NS:![0-9]+]] = !MDNamespace(name: "B", scope: [[CTXT:![0-9]+]], file: [[FOOCPP]], line: 1)
// CHECK: [[CTXT]] = !MDNamespace(name: "A", scope: null, file: [[FILE:![0-9]+]], line: 5)
// CHECK: [[FILE]] = !MDFile(filename: "{{.*}}debug-info-namespace.cpp",
// CHECK: [[BAR:![0-9]+]] = !MDCompositeType(tag: DW_TAG_structure_type, name: "bar",
// CHECK-SAME:                               line: 6
// CHECK-SAME:                               DIFlagFwdDecl
// CHECK: [[F1:![0-9]+]] = !MDSubprogram(name: "f1",{{.*}} line: 4
// CHECK-SAME:                           isDefinition: true
// CHECK: [[FUNC:![0-9]+]] = !MDSubprogram(name: "func",{{.*}} isDefinition: true
// CHECK: [[FUNC_FWD:![0-9]+]] = !MDSubprogram(name: "func_fwd",{{.*}} line: 47,{{.*}} isDefinition: true
// CHECK: [[I:![0-9]+]] = !MDGlobalVariable(name: "i",{{.*}} scope: [[NS]],
// CHECK: [[VAR_FWD:![0-9]+]] = !MDGlobalVariable(name: "var_fwd",{{.*}} scope: [[NS]],
// CHECK-SAME:                                    line: 44
// CHECK-SAME:                                    isDefinition: true

// CHECK: [[MODULES]] = !{[[M1:![0-9]+]], [[M2:![0-9]+]], [[M3:![0-9]+]], [[M4:![0-9]+]], [[M5:![0-9]+]], [[M6:![0-9]+]], [[M7:![0-9]+]], [[M8:![0-9]+]], [[M9:![0-9]+]], [[M10:![0-9]+]], [[M11:![0-9]+]], [[M12:![0-9]+]], [[M13:![0-9]+]], [[M14:![0-9]+]], [[M15:![0-9]+]], [[M16:![0-9]+]], [[M17:![0-9]+]]}
// CHECK: [[M1]] = !MDImportedEntity(tag: DW_TAG_imported_module, scope: [[CTXT]], entity: [[NS]], line: 15)
// CHECK: [[M2]] = !MDImportedEntity(tag: DW_TAG_imported_module, scope: [[CU]], entity: [[CTXT]],
// CHECK: [[M3]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, name: "E", scope: [[CU]], entity: [[CTXT]], line: 19)
// CHECK: [[M4]] = !MDImportedEntity(tag: DW_TAG_imported_module, scope: [[LEX2:![0-9]+]], entity: [[NS]], line: 23)
// CHECK: [[LEX2]] = distinct !MDLexicalBlock(scope: [[LEX1:![0-9]+]], file: [[FOOCPP]],
// CHECK: [[LEX1]] = distinct !MDLexicalBlock(scope: [[FUNC]], file: [[FOOCPP]],
// CHECK: [[M5]] = !MDImportedEntity(tag: DW_TAG_imported_module, scope: [[FUNC]], entity: [[CTXT]],
// CHECK: [[M6]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FOO:!"_ZTSN1A1B3fooE"]], line: 27)
// CHECK: [[M7]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[BAR:!"_ZTSN1A1B3barE"]]
// CHECK: [[M8]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[F1]]
// CHECK: [[M9]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[I]]
// CHECK: [[M10]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[BAZ:![0-9]+]]
// CHECK: [[BAZ]] = !MDDerivedType(tag: DW_TAG_typedef, name: "baz", scope: [[NS]], file: [[FOOCPP]],
// CHECK-SAME:                     baseType: !"_ZTSN1A1B3barE"
// CHECK: [[M11]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, name: "X", scope: [[FUNC]], entity: [[CTXT]]
// CHECK: [[M12]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, name: "Y", scope: [[FUNC]], entity: [[M11]]
// CHECK: [[M13]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[VAR_DECL:![0-9]+]]
// CHECK: [[VAR_DECL]] = !MDGlobalVariable(name: "var_decl", linkageName: "_ZN1A1B8var_declE", scope: [[NS]],{{.*}} line: 8,
// CHECK: [[M14]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FUNC_DECL:![0-9]+]]
// CHECK: [[FUNC_DECL]] = !MDSubprogram(name: "func_decl",
// CHECK-SAME:                          scope: [[NS]], file: [[FOOCPP]], line: 9
// CHECK: [[M15]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[VAR_FWD:![0-9]+]]
// CHECK: [[M16]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FUNC_FWD:![0-9]+]]
// CHECK: [[M17]] = !MDImportedEntity(tag: DW_TAG_imported_declaration, scope: [[CTXT]], entity: [[I]]

// CHECK-GMLT: [[CU:![0-9]+]] = !MDCompileUnit(
// CHECK-GMLT-SAME:                            emissionKind: 2,
// CHECK-GMLT-SAME:                            imports: [[MODULES:![0-9]+]]
// CHECK-GMLT: [[MODULES]] = !{}

// CHECK-NOLIMIT: !MDCompositeType(tag: DW_TAG_structure_type, name: "bar",{{.*}} line: 6,
// CHECK-NOLIMIT-NOT:              DIFlagFwdDecl
// CHECK-NOLIMIT-SAME:             ){{$}}

// REQUIRES: shell-preserves-root
// REQUIRES: dw2
