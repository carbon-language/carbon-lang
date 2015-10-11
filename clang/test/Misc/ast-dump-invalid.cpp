// RUN: not %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fms-extensions -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix CHECK -strict-whitespace %s

namespace TestInvalidRParenOnCXXUnresolvedConstructExpr {
template <class T>
void f(T i, T j) {
  return T (i, j;
}
}

// CHECK: NamespaceDecl {{.*}} <{{.*}}> {{.*}} TestInvalidRParenOnCXXUnresolvedConstructExpr
// CHECK-NEXT: `-FunctionTemplateDecl
// CHECK-NEXT:   |-TemplateTypeParmDecl
// CHECK-NEXT:   `-FunctionDecl
// CHECK-NEXT:     |-ParmVarDecl
// CHECK-NEXT:     |-ParmVarDecl
// CHECK-NEXT:     `-CompoundStmt
// CHECK-NEXT:       `-ReturnStmt
// CHECK-NEXT:         `-CXXUnresolvedConstructExpr {{.*}} <col:10, col:16> 'T'
// CHECK-NEXT:           |-DeclRefExpr {{.*}} <col:13> 'T' lvalue ParmVar {{.*}} 'i' 'T'
// CHECK-NEXT:           `-DeclRefExpr {{.*}} <col:16> 'T' lvalue ParmVar {{.*}} 'j' 'T'


namespace TestInvalidIf {
int g(int i) {
  if (invalid_condition)
    return 4;
  else
    return i;
}
}
// CHECK: NamespaceDecl {{.*}} <{{.*}}> {{.*}} TestInvalidIf
// CHECK-NEXT: `-FunctionDecl
// CHECK-NEXT:   |-ParmVarDecl
// CHECK-NEXT:   `-CompoundStmt
// CHECK-NEXT:     `-IfStmt {{.*}} <line:25:3, line:28:12>
// CHECK-NEXT:       |-<<<NULL>>>
// CHECK-NEXT:       |-OpaqueValueExpr {{.*}} <<invalid sloc>> '_Bool'
// CHECK-NEXT:       |-ReturnStmt {{.*}} <line:26:5, col:12>
// CHECK-NEXT:       | `-IntegerLiteral {{.*}} <col:12> 'int' 4
// CHECK-NEXT:       `-ReturnStmt {{.*}} <line:28:5, col:12>
// CHECK-NEXT:         `-ImplicitCastExpr {{.*}} <col:12> 'int' <LValueToRValue>
// CHECK-NEXT:           `-DeclRefExpr {{.*}} <col:12> 'int' lvalue ParmVar {{.*}} 'i' 'int'

