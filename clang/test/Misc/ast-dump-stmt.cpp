// RUN: %clang_cc1 -fcxx-exceptions -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

namespace n {
void function() {}
int Variable;
}
using n::function;
using n::Variable;
void TestFunction() {
  void (*f)() = &function;
// CHECK:       DeclRefExpr{{.*}} (UsingShadow{{.*}}function
  Variable = 4;
// CHECK:       DeclRefExpr{{.*}} (UsingShadow{{.*}}Variable
}

// CHECK: FunctionDecl {{.*}} TestCatch1
void TestCatch1() {
// CHECK:       CXXTryStmt
// CHECK-NEXT:    CompoundStmt
  try {
  }
// CHECK-NEXT:    CXXCatchStmt
// CHECK-NEXT:      VarDecl {{.*}} x
// CHECK-NEXT:      CompoundStmt
  catch (int x) {
  }
}

// CHECK: FunctionDecl {{.*}} TestCatch2
void TestCatch2() {
// CHECK:       CXXTryStmt
// CHECK-NEXT:    CompoundStmt
  try {
  }
// CHECK-NEXT:    CXXCatchStmt
// CHECK-NEXT:      NULL
// CHECK-NEXT:      CompoundStmt
  catch (...) {
  }
}
