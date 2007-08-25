// RUN: clang -parse-ast-check %s
_Complex double X;
void test1(int c) {
  X = 5;
}
void test2() {
  int i;
  double d = i;
  double _Complex a = 5;

  test1(a);
  a = 5;
  d = i;
}
int test3() {
  int a[2];
  a[0] = test3; // expected-warning{{incompatible types assigning 'int (void)' to 'int'}}
}
short x; void test4(char c) { x += c; }
int y; void test5(char c) { y += c; }
