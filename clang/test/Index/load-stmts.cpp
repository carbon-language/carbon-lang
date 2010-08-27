typedef int T;
struct X { int a, b; };
void f(int x) {
  for (T y = x; T z = x; ++x) {
  }
  if (T *z2 = &x) { }
  while (T *z3 = &x) { }
  switch (T z4 = x) {
  case 17: break;
  }
}

// Test handling of C++ base specifiers.
class A {
  void doA();
};

class B {
  void doB();
};

class C : public A, private B {
  void doC();
};

class D : virtual public C, virtual private A {};

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-stmts.cpp:1:13: TypedefDecl=T:1:13 (Definition) Extent=[1:13 - 1:14]
// CHECK: load-stmts.cpp:2:8: StructDecl=X:2:8 (Definition) Extent=[2:1 - 2:23]
// CHECK: load-stmts.cpp:2:16: FieldDecl=a:2:16 (Definition) Extent=[2:16 - 2:17]
// CHECK: load-stmts.cpp:2:19: FieldDecl=b:2:19 (Definition) Extent=[2:19 - 2:20]
// CHECK: load-stmts.cpp:3:6: FunctionDecl=f:3:6 (Definition) Extent=[3:6 - 11:2]
// CHECK: load-stmts.cpp:3:12: ParmDecl=x:3:12 (Definition) Extent=[3:8 - 3:13]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[3:15 - 11:2]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[4:3 - 5:4]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[4:8 - 4:16]
// CHECK: load-stmts.cpp:4:10: VarDecl=y:4:10 (Definition) Extent=[4:8 - 4:15]
// CHECK: load-stmts.cpp:4:8: TypeRef=T:1:13 Extent=[4:8 - 4:9]
// CHECK: load-stmts.cpp:4:14: DeclRefExpr=x:3:12 Extent=[4:14 - 4:15]
// CHECK: load-stmts.cpp:4:19: VarDecl=z:4:19 (Definition) Extent=[4:17 - 4:24]
// CHECK: load-stmts.cpp:4:17: TypeRef=T:1:13 Extent=[4:17 - 4:18]
// CHECK: load-stmts.cpp:4:23: DeclRefExpr=x:3:12 Extent=[4:23 - 4:24]
// CHECK: load-stmts.cpp:4:19: UnexposedExpr=z:4:19 Extent=[4:19 - 4:20]
// CHECK: load-stmts.cpp:4:19: DeclRefExpr=z:4:19 Extent=[4:19 - 4:20]
// CHECK: load-stmts.cpp:4:26: UnexposedExpr= Extent=[4:26 - 4:29]
// CHECK: load-stmts.cpp:4:28: DeclRefExpr=x:3:12 Extent=[4:28 - 4:29]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[4:31 - 5:4]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[6:3 - 6:22]
// CHECK: load-stmts.cpp:6:10: VarDecl=z2:6:10 (Definition) Extent=[6:7 - 6:17]
// CHECK: load-stmts.cpp:6:7: TypeRef=T:1:13 Extent=[6:7 - 6:8]
// CHECK: load-stmts.cpp:6:15: UnexposedExpr= Extent=[6:15 - 6:17]
// CHECK: load-stmts.cpp:6:16: DeclRefExpr=x:3:12 Extent=[6:16 - 6:17]
// CHECK: load-stmts.cpp:6:10: UnexposedExpr=z2:6:10 Extent=[6:10 - 6:12]
// CHECK: load-stmts.cpp:6:10: DeclRefExpr=z2:6:10 Extent=[6:10 - 6:12]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[6:19 - 6:22]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[7:3 - 7:25]
// CHECK: load-stmts.cpp:7:13: VarDecl=z3:7:13 (Definition) Extent=[7:10 - 7:20]
// CHECK: load-stmts.cpp:7:10: TypeRef=T:1:13 Extent=[7:10 - 7:11]
// CHECK: load-stmts.cpp:7:18: UnexposedExpr= Extent=[7:18 - 7:20]
// CHECK: load-stmts.cpp:7:19: DeclRefExpr=x:3:12 Extent=[7:19 - 7:20]
// CHECK: load-stmts.cpp:7:13: UnexposedExpr=z3:7:13 Extent=[7:13 - 7:15]
// CHECK: load-stmts.cpp:7:13: DeclRefExpr=z3:7:13 Extent=[7:13 - 7:15]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[7:22 - 7:25]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[8:3 - 10:4]
// CHECK: load-stmts.cpp:8:13: VarDecl=z4:8:13 (Definition) Extent=[8:11 - 8:19]
// CHECK: load-stmts.cpp:8:11: TypeRef=T:1:13 Extent=[8:11 - 8:12]
// CHECK: load-stmts.cpp:8:18: DeclRefExpr=x:3:12 Extent=[8:18 - 8:19]
// CHECK: load-stmts.cpp:8:13: DeclRefExpr=z4:8:13 Extent=[8:13 - 8:15]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[8:21 - 10:4]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[9:3 - 9:17]
// CHECK: load-stmts.cpp:9:8: UnexposedExpr= Extent=[9:8 - 9:10]
// CHECK: <invalid loc>:0:0: UnexposedStmt= Extent=[9:12 - 9:17]
// CHECK: load-stmts.cpp:14:7: ClassDecl=A:14:7 (Definition) Extent=[14:1 - 16:2]
// CHECK: load-stmts.cpp:15:8: CXXMethod=doA:15:8 Extent=[15:8 - 15:13]
// CHECK: load-stmts.cpp:18:7: ClassDecl=B:18:7 (Definition) Extent=[18:1 - 20:2]
// CHECK: load-stmts.cpp:19:8: CXXMethod=doB:19:8 Extent=[19:8 - 19:13]
// CHECK: load-stmts.cpp:22:7: ClassDecl=C:22:7 (Definition) Extent=[22:1 - 24:2]
// CHECK: <invalid loc>:0:0: C++ base class specifier=class A:14:7 [access=public isVirtual=false]
// CHECK: <invalid loc>:0:0: C++ base class specifier=class B:18:7 [access=private isVirtual=false]
// CHECK: load-stmts.cpp:23:8: CXXMethod=doC:23:8 Extent=[23:8 - 23:13]
// CHECK: load-stmts.cpp:26:7: ClassDecl=D:26:7 (Definition) Extent=[26:1 - 26:49]
// CHECK: <invalid loc>:0:0: C++ base class specifier=class C:22:7 [access=public isVirtual=true]
// CHECK: <invalid loc>:0:0: C++ base class specifier=class A:14:7 [access=private isVirtual=true]

