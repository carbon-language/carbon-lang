// RUN: c-index-test -test-print-linkage-source %s | FileCheck %s

enum Baz { Qux = 0 };
int x;
void foo();
static int w;
void bar(int y) {
  static int z;
  int k;
}
extern int n;

// CHECK: EnumDecl=Baz:3:6 (Definition)linkage=External
// CHECK: EnumConstantDecl=Qux:3:12 (Definition)linkage=External
// CHECK: VarDecl=x:4:5linkage=External
// CHECK: FunctionDecl=foo:5:6linkage=External
// CHECK: VarDecl=w:6:12linkage=External
// CHECK: FunctionDecl=bar:7:6 (Definition)linkage=External
// CHECK: ParmDecl=y:7:14 (Definition)linkage=External
// CHECK: VarDecl=z:8:14 (Definition)linkage=External
// CHECK: VarDecl=k:9:7 (Definition)linkage=External
// CHECK: VarDecl=n:11:12

