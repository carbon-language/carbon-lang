// RUN: clang -emit-llvm %s

int main(void)
{
  double _Complex a = 5;
  double _Complex b = 42;

  return a * b != b * a;
}

_Complex double bar(int);
void test(_Complex double*);
void takecomplex(_Complex double);

void test2(int c) {
  _Complex double X;
  X = bar(1);
  test(&X);
  takecomplex(X);
}

_Complex double g1, g2;
_Complex float cf;
double D;

void test3() {
  g1 = g1 + g2;
  g1 = g1 - g2;
  g1 = g1 * g2;
  g1 = +-~g1;

  double Gr = __real g1;

  //cf += D;  // fixme: sema bug
  D += cf;
  cf /= g1;
}
