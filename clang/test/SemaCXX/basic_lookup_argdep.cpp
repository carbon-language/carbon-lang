// RUN: clang -fsyntax-only -verify %s

namespace N {
  struct X { };
  
  X operator+(X, X);

  void f(X);
  void g(X);

  void test_multiadd(X x) {
    (void)(x + x);
  }
}

namespace M {
  struct Y : N::X { };
}

void f();

void test_operator_adl(N::X x, M::Y y) {
  (void)(x + x);
  (void)(y + y);
}

void test_func_adl(N::X x, M::Y y) {
  f(x);
  f(y);
  (f)(x); // expected-error{{too many arguments to function call}}
  ::f(x); // expected-error{{too many arguments to function call}}
}

namespace N {
  void test_multiadd2(X x) {
    (void)(x + x);
  }
}


void test_func_adl_only(N::X x) {
  // FIXME: here, despite the fact that the name lookup for 'g' fails,
  // this is well-formed code. The fix will go into Sema::ActOnCallExpr.
  //  g(x);
}
