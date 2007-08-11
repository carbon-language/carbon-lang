// RUN: clang -emit-llvm %s
// XFAIL: *

int main(void)
{
  double _Complex a = 5;
  double _Complex b = 42;

  return a * b != b * a;
}

_Complex double bar(int);
void test(_Complex double*);

void test2(int c) {
  _Complex double X;
  X = bar(1);
  test(&X);
}

