template<typename ...Args>
int f(Args ...args) {
  return sizeof...(args) + sizeof...(Args);
}

void test() {
  int a;
  decltype(a) b;
}

// RUN: c-index-test -test-annotate-tokens=%s:1:1:5:1 -std=c++0x %s | FileCheck %s
// CHECK: Identifier: "args" [3:20 - 3:24] UnexposedExpr=args:2:15
// CHECK: Identifier: "Args" [3:38 - 3:42] TypeRef=Args:1:22

// RUN: c-index-test -test-annotate-tokens=%s:8:1:9:1 -std=c++0x %s | FileCheck -check-prefix=CHECK-DECLTYPE %s
// CHECK-DECLTYPE: Identifier: "a" [8:12 - 8:13] DeclRefExpr=a:7:7
