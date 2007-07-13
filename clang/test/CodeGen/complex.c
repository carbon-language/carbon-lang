// RUN: clang -emit-llvm %s
// XFAIL: *

void main(void)
{
  double _Complex a = 5;
  double _Complex b = 42;

  return a * b == b * a;
}
