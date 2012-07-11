// RUN: %clang_cc1 -fsyntax-only -fcolor-diagnostics %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fcolor-diagnostics -fdiagnostics-show-template-tree %s 2>&1 | FileCheck %s -check-prefix=TREE
// REQUIRES: ansi-escape-sequences
template<typename> struct foo {};
void func(foo<int>);
int main() {
  func(foo<double>());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'foo<{{.}}[0;1;36mdouble{{.}}[0m>' to 'foo<{{.}}[0;1;36mint{{.}}[0m>' for 1st argument{{.}}[0m
// TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// TREE:  foo<
// TREE:    [{{.}}[0;1;36mdouble{{.}}[0m != {{.}}[0;1;36mint{{.}}[0m]>{{.}}[0m

foo<int> A;
foo<double> &B = A;
// CHECK: {{.*}}non-const lvalue reference to type 'foo<{{.}}[0;1;36mdouble{{.}}[0m{{.}}[1m>' cannot bind to a value of unrelated type 'foo<{{.}}[0;1;36mint{{.}}[0m{{.}}[1m>'{{.}}[0m
// TREE: non-const lvalue reference cannot bind to a value of unrelated type
// TREE:   foo<
// TREE:     [{{.}}[0;1;36mdouble{{.}}[0m{{.}}[1m != {{.}}[0;1;36mint{{.}}[0m{{.}}[1m]>{{.}}[0m
