// RUN: %clang_cc1 -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

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
