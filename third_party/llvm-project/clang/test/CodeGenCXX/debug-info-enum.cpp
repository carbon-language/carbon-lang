// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -gcodeview -debug-info-kind=limited %s -o - | FileCheck --check-prefix MSVC %s

// CHECK: !DICompileUnit(
// CHECK-SAME:           enums: [[ENUMS:![0-9]*]]
// CHECK: [[ENUMS]] = !{[[E1:![0-9]*]], [[E2:![0-9]*]], [[E3:![0-9]*]]}

// In MSVC check that used enum values are emitted as globals.
// MSVC: !DICompileUnit(
// MSVC-SAME:           globals: [[GLOBALS:![0-9]*]]
// MSVC: [[GLOBALS]] = !{[[G1:![0-9]*]], [[G2:![0-9]*]]}

namespace test1 {
// CHECK: [[E1]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e"
// CHECK-SAME:                      scope: [[TEST1:![0-9]*]]
// CHECK-SAME:                      elements: [[TEST1_ENUMS:![0-9]*]]
// CHECK-SAME:                      identifier: "_ZTSN5test11eE"
// CHECK: [[TEST1]] = !DINamespace(name: "test1"
// CHECK: [[TEST1_ENUMS]] = !{[[TEST1_E:![0-9]*]]}
// CHECK: [[TEST1_E]] = !DIEnumerator(name: "E", value: 0, isUnsigned: true)

// MSVC: [[G1]] = !DIGlobalVariableExpression(var: [[VAR1:![0-9]*]],
// MSVC-SAME:                                 expr: !DIExpression(DW_OP_constu, 0
// MSVC: [[VAR1]] = distinct !DIGlobalVariable(name: "E"
enum e { E };
void foo() {
  int v = E;
}
}

namespace test2 {
// rdar://8195980
// CHECK: [[E2]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e"
// CHECK-SAME:                      scope: [[TEST2:![0-9]+]]
// CHECK-SAME:                      elements: [[TEST1_ENUMS]]
// CHECK-SAME:                      identifier: "_ZTSN5test21eE"
// CHECK: [[TEST2]] = !DINamespace(name: "test2"

// MSVC: [[G2]] = !DIGlobalVariableExpression(var: [[VAR2:![0-9]*]],
// MSVC-SAME:                                 expr: !DIExpression(DW_OP_constu, 0
// MSVC: [[VAR2]] = distinct !DIGlobalVariable(name: "E"
enum e { E };
bool func(int i) {
  return i == E;
}
}

namespace test3 {
// CHECK: [[E3]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e"
// CHECK-SAME:                      scope: [[TEST3:![0-9]*]]
// CHECK-SAME:                      elements: [[TEST3_ENUMS:![0-9]*]]
// CHECK-SAME:                      identifier: "_ZTSN5test31eE"
// CHECK: [[TEST3]] = !DINamespace(name: "test3"
// CHECK: [[TEST3_ENUMS]] = !{[[TEST3_E:![0-9]*]]}
// CHECK: [[TEST3_E]] = !DIEnumerator(name: "E", value: -1)
enum e { E = -1 };
void func() {
  e x;
}
}

namespace test4 {
// Don't try to build debug info for a dependent enum.
// CHECK-NOT: test4
template <typename T>
struct S {
  enum e { E = T::v };
};
}
