// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: SwitchStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: BreakStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 4
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: BreakStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: BreakStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: DefaultStmt
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: NullStmt

void expr() {
  f();
}
