// RUN: %clang_cc1 -std=c++11 -verify %s

typedef int (*fp)(int);
int surrogate(int);

struct X {
  X() = default;  // expected-note{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  X(const X&) = default;  // expected-note{{candidate constructor not viable: no known conversion from 'bool' to 'const X' for 1st argument}}
  X(bool b) __attribute__((enable_if(b, "chosen when 'b' is true")));  // expected-note{{candidate disabled: chosen when 'b' is true}}

  void f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));
  void f(int n) __attribute__((enable_if(n == 1, "chosen when 'n' is one")));  // expected-note{{member declaration nearly matches}} expected-note{{candidate disabled: chosen when 'n' is one}}

  static void s(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));  // expected-note2{{candidate disabled: chosen when 'n' is zero}}

  void conflict(int n) __attribute__((enable_if(n+n == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}
  void conflict(int n) __attribute__((enable_if(n*2 == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}

  operator long() __attribute__((enable_if(true, "chosen on your platform")));
  operator int() __attribute__((enable_if(false, "chosen on other platform")));

  operator fp() __attribute__((enable_if(false, "never enabled"))) { return surrogate; }  // expected-note{{conversion candidate of type 'int (*)(int)'}}  // FIXME: the message is not displayed
};

void X::f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")))  // expected-note{{member declaration nearly matches}} expected-note{{candidate disabled: chosen when 'n' is zero}}
{
}

void X::f(int n) __attribute__((enable_if(n == 2, "chosen when 'n' is two")))  // expected-error{{out-of-line definition of 'f' does not match any declaration in 'X'}} expected-note{{candidate disabled: chosen when 'n' is two}}
{
}

X x1(true);
X x2(false); // expected-error{{no matching constructor for initialization of 'X'}}

__attribute__((deprecated)) constexpr int old() { return 0; }  // expected-note2{{'old' has been explicitly marked deprecated here}}
void deprec1(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}
void deprec2(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}

void overloaded(int);
void overloaded(long);

struct Nothing { };
template<typename T> void typedep(T t) __attribute__((enable_if(t, "")));  // expected-note{{candidate disabled:}}  expected-error{{value of type 'Nothing' is not contextually convertible to 'bool'}}
template<int N> void valuedep() __attribute__((enable_if(N == 1, "")));

// FIXME: we skip potential constant expression evaluation on value dependent
// enable-if expressions
int not_constexpr();
template<int N> void valuedep() __attribute__((enable_if(N == not_constexpr(), "")));

template <typename T> void instantiationdep() __attribute__((enable_if(sizeof(sizeof(T)) != 0, "")));

void test() {
  X x;
  x.f(0);
  x.f(1);
  x.f(2);  // no error, suppressed by erroneous out-of-line definition
  x.f(3);  // expected-error{{no matching member function for call to 'f'}}

  x.s(0);
  x.s(1);  // expected-error{{no matching member function for call to 's'}}

  X::s(0);
  X::s(1);  // expected-error{{no matching member function for call to 's'}}

  x.conflict(5);  // expected-error{{call to member function 'conflict' is ambiguous}}

  deprec2(0);

  overloaded(x);

  int i = x(1);  // expected-error{{no matching function for call to object of type 'X'}}

  Nothing n;
  typedep(0);  // expected-error{{no matching function for call to 'typedep'}}
  typedep(1);
  typedep(n);  // expected-note{{in instantiation of function template specialization 'typedep<Nothing>' requested here}}
}

template <typename T> class C {
  void f() __attribute__((enable_if(T::expr == 0, ""))) {}
  void g() { f(); }
};

int fn3(bool b) __attribute__((enable_if(b, "")));
template <class T> void test3() {
  fn3(sizeof(T) == 1);
}

// FIXME: issue an error (without instantiation) because ::h(T()) is not
// convertible to bool, because return types aren't overloadable.
void h(int);
template <typename T> void outer() {
  void local_function() __attribute__((enable_if(::h(T()), "")));
  local_function();
};

namespace PR20988 {
  struct Integer {
    Integer(int);
  };

  int fn1(const Integer &) __attribute__((enable_if(true, "")));
  template <class T> void test1() {
    int &expr = T::expr();
    fn1(expr);
  }

  int fn2(const Integer &) __attribute__((enable_if(false, "")));  // expected-note{{candidate disabled}}
  template <class T> void test2() {
    int &expr = T::expr();
    fn2(expr);  // expected-error{{no matching function for call to 'fn2'}}
  }

  int fn3(bool b) __attribute__((enable_if(b, "")));
  template <class T> void test3() {
    fn3(sizeof(T) == 1);
  }
}
