// RxUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump %s > /dev/null

template <int X, typename Y, int Z = 5>
struct foo {
  int constant;
  foo() {}
  Y getSum() { return Y(X + Z); }
};

template <int A, typename B>
B bar() {
  return B(A);
}

void baz() {
  int x = bar<5, int>();
  int y = foo<5, int>().getSum();
  double z = foo<2, double, 3>().getSum();
}

// Template instantiation - foo
// CHECK: template <int X = 5, typename Y = int, int Z = 5> struct foo {
// CHECK: template <int X = 2, typename Y = double, int Z = 3> struct foo {

// Template definition - foo
// CHECK: template <int X, typename Y, int Z = (IntegerLiteral {{.*}} 'int' 5)

// Template instantiation - bar
// CHECK: template <int A = 5, typename B = int> int bar()

// Template definition - bar
// CHECK: template <int A, typename B> B bar()
