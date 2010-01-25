typedef int T;
struct X { int a, b; };
void f(void *ptr) {
  T* t_ptr = (T *)ptr;
  (void)sizeof(T);
  struct X x = (struct X){1, 2};
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s

// CHECK: load-exprs.c:4:15: TypeRef=T:1:13 [Extent=4:15:4:15]
// CHECK: load-exprs.c:5:16: TypeRef=T:1:13 [Extent=5:16:5:16]
// CHECK: load-exprs.c:6:10: TypeRef=struct X:2:8 [Extent=6:10:6:10]
// CHECK: load-exprs.c:6:24: TypeRef=struct X:2:8 [Extent=6:24:6:24]
