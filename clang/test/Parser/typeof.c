// RUN: clang -parse-ast-check %s -pedantic

typedef int TInt;

static void test() {
  int *pi;

  typeof(TInt) anInt; // expected-warning{{extension used}}
  typeof(const int) aci; // expected-warning{{extension used}}
  const typeof (*pi) aConstInt; // expected-warning{{extension used}}
  int xx;
  short typeof (*pi) aShortInt; // expected-error{{'short typeof' is invalid}}
  int *i;
  i = aci; // expected-warning{{incompatible types assigning 'typeof(int const)' to 'int *'}}
  i = anInt; // expected-warning{{incompatible types assigning 'typeof(TInt)' to 'int *'}}
  i = aConstInt; // expected-warning{{incompatible types assigning 'typeof(<expr>) const' to 'int *'}}
  i = xx; // expected-warning{{incompatible types assigning 'int' to 'int *'}}
}
