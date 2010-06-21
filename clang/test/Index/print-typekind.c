typedef int FooType;
int *p;
int *f(int *p, char *x, FooType z) {
  FooType w = z;
  return p + z;
}

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: TypedefDecl=FooType:1:13 (Definition) typekind=Typedef [canonical=Int]
// CHECK: VarDecl=p:2:6 typekind=Pointer
// CHECK: FunctionDecl=f:3:6 (Definition) typekind=FunctionProto [canonical=FunctionProto] [result=Pointer]
// CHECK: ParmDecl=p:3:13 (Definition) typekind=Pointer
// CHECK: ParmDecl=x:3:22 (Definition) typekind=Pointer
// CHECK: ParmDecl=z:3:33 (Definition) typekind=Typedef [canonical=Int]
// CHECK: VarDecl=w:4:11 (Definition) typekind=Typedef [canonical=Int]
// CHECK: DeclRefExpr=z:3:33 typekind=Typedef [canonical=Int]
// CHECK: UnexposedExpr= typekind=Pointer
// CHECK: DeclRefExpr=p:3:13 typekind=Pointer
// CHECK: DeclRefExpr=z:3:33 typekind=Typedef [canonical=Int]

