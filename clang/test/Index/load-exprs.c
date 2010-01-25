typedef int T;
struct X { int a, b; };
void f(void *ptr) {
  T* t_ptr = (T *)ptr;
  (void)sizeof(T);
  struct X x = (struct X){1, 2};
  void *xx = ptr ? : &x;
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s

// CHECK: load-exprs.c:4:15: TypeRef=T:1:13 [Extent=4:15:4:15]
// CHECK: load-exprs.c:5:16: TypeRef=T:1:13 [Extent=5:16:5:16]
// CHECK: load-exprs.c:6:10: TypeRef=struct X:2:8 [Extent=6:10:6:10]
// CHECK: load-exprs.c:6:24: TypeRef=struct X:2:8 [Extent=6:24:6:24]
// CHECK: load-exprs.c:7:9: VarDecl=xx:7:9 (Definition) [Extent=7:3:7:23]
// CHECK: load-exprs.c:7:14: DeclRefExpr=ptr:3:14 [Extent=7:14:7:16]
// CHECK: load-exprs.c:7:23: DeclRefExpr=x:6:12 [Extent=7:23:7:23]
