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
