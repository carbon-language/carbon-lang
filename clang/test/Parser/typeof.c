// RUN: clang -parse-ast-check %s

typedef int TInt;

static void test() {
  int *pi;

  int typeof (int) aIntInt; // expected-error{{cannot combine with previous 'int' declaration specifier}}
  short typeof (int) aShortInt; // expected-error{{'short typeof' is invalid}} 
  int int ttt; // expected-error{{cannot combine with previous 'int' declaration specifier}}
  typeof(TInt) anInt; 
  short TInt eee; // expected-error{{parse error}}
  void ary[7] fff; // expected-error{{array has incomplete element type 'void'}} expected-error{{parse error}}
  typeof(void ary[7]) anIntError; // expected-error{{expected ')'}} expected-error{{to match this '('}}
  typeof(const int) aci; 
  const typeof (*pi) aConstInt; 
  int xx;
  int *i;
  i = aci; // expected-warning{{incompatible types assigning 'typeof(int const)' to 'int *'}}
  i = anInt; // expected-warning{{incompatible types assigning 'typeof(TInt)' to 'int *'}}
  i = aConstInt; // expected-warning{{incompatible types assigning 'typeof(*pi) const' to 'int *'}}
  i = xx; // expected-warning{{incompatible types assigning 'int' to 'int *'}}
}
