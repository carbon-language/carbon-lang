// RUN: %clang_cc1 -std=c++1y -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

struct S {
  // dummy ctor to make this a literal type
  constexpr S(int);

  S();

  int arr[10];

  constexpr int &get(int n) { return arr[n]; }
  constexpr const int &get(int n) const { return arr[n]; }
};

S s = S();
const S &sr = s;
static_assert(&s.get(4) - &sr.get(2) == 2, "");

// Compound-statements can be used in constexpr functions.
constexpr int e() {{{{}} return 5; }}
static_assert(e() == 5, "");

// Types can be defined in constexpr functions.
constexpr int f() {
  enum E { e1, e2, e3 };

  struct S {
    constexpr S(E e) : e(e) {}
    constexpr int get() { return e; }
    E e;
  };

  return S(e2).get();
}
static_assert(f() == 1, "");

// Variables can be declared in constexpr functions.
constexpr int g(int k) {
  const int n = 9;
  int k2 = k * k;
  int k3 = k2 * k;
  return 3 * k3 + 5 * k2 + n * k - 20;
}
static_assert(g(2) == 42, "");
constexpr int h(int n) {
  static const int m = n; // expected-error {{static variable not permitted in a constexpr function}}
  return m;
}
constexpr int i(int n) {
  thread_local const int m = n; // expected-error {{thread_local variable not permitted in a constexpr function}}
  return m;
}

// if-statements can be used in constexpr functions.
constexpr int j(int k) {
  if (k == 5)
    return 1;
  if (k == 1)
    return 5;
  else {
    if (int n = 2 * k - 4) {
      return n + 1;
      return 2;
    }
  }
} // expected-note 2{{control reached end of constexpr function}}
static_assert(j(0) == -3, "");
static_assert(j(1) == 5, "");
static_assert(j(2), ""); // expected-error {{constant expression}} expected-note {{in call to 'j(2)'}}
static_assert(j(3) == 3, "");
static_assert(j(4) == 5, "");
static_assert(j(5) == 1, "");

// There can be 0 return-statements.
constexpr void k() {
}

// If the return type is not 'void', no return statements => never a constant
// expression, so still diagnose that case.
[[noreturn]] constexpr int fn() { // expected-error {{no return statement in constexpr function}}
  fn();
}

// We evaluate the body of a constexpr constructor, to check for side-effects.
struct U {
  constexpr U(int n) {
    if (j(n)) {} // expected-note {{in call to 'j(2)'}}
  }
};
constexpr U u1{1};
constexpr U u2{2}; // expected-error {{constant expression}} expected-note {{in call to 'U(2)'}}

// We allow expression-statements.
constexpr int l(bool b) {
  if (b)
    throw "invalid value for b!"; // expected-note {{subexpression not valid}}
  return 5;
}
static_assert(l(false) == 5, "");
static_assert(l(true), ""); // expected-error {{constant expression}} expected-note {{in call to 'l(true)'}}

// Potential constant expression checking is still applied where possible.
constexpr int htonl(int x) { // expected-error {{never produces a constant expression}}
  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int maybe_htonl(bool isBigEndian, int x) {
  if (isBigEndian)
    return x;

  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int swapped = maybe_htonl(false, 123); // expected-error {{constant expression}} expected-note {{in call}}

namespace NS {
  constexpr int n = 0;
}
constexpr int namespace_alias() {
  namespace N = NS;
  return N::n;
}

namespace assign {
  constexpr int a = 0;
  const int b = 0;
  int c = 0; // expected-note 2{{here}}

  constexpr void set(const int &a, int b) {
    const_cast<int&>(a) = b; // expected-note 2{{constant expression cannot modify an object that is visible outside that expression}}
  }
  constexpr int wrap(int a, int b) {
    set(a, b);
    return a;
  }

  static_assert((set(a, 1), a) == 1, ""); // expected-error {{constant expression}} expected-note {{in call to 'set(a, 1)'}}
  static_assert((set(b, 1), b) == 1, ""); // expected-error {{constant expression}} expected-note {{in call to 'set(b, 1)'}}
  static_assert((set(c, 1), c) == 1, ""); // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}

  static_assert(wrap(a, 1) == 1, "");
  static_assert(wrap(b, 1) == 1, "");
  static_assert(wrap(c, 1) == 1, ""); // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
}

namespace string_assign {
  template<typename T>
  constexpr void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
  template<typename Iterator>
  constexpr void reverse(Iterator begin, Iterator end) {
#if 0 // FIXME: once implementation is complete...
    while (begin != end && begin != --end)
      swap(*begin++, *end);
#else
    if (begin != end) {
      end = end - 1;
      if (begin == end)
        return;
      swap(*begin, *end);
      begin = begin + 1;
      reverse(begin, end);
    }
#endif
  }
  template<typename Iterator1, typename Iterator2>
  constexpr bool equal(Iterator1 a, Iterator1 ae, Iterator2 b, Iterator2 be) {
#if 0 // FIXME: once implementation is complete...
    while (a != ae && b != be) {
      if (*a != *b)
        return false;
      ++a, ++b;
    }
#else
    if (a != ae && b != be) {
      if (*a != *b)
        return false;
      a = a + 1;
      b = b + 1;
      return equal(a, ae, b, be);
    }
#endif
    return a == ae && b == be;
  }
  constexpr bool test1(int n) {
    char stuff[100] = "foobarfoo";
    const char stuff2[100] = "oofraboof";
    reverse(stuff, stuff + n); // expected-note {{cannot refer to element 101 of array of 100 elements}}
    return equal(stuff, stuff + n, stuff2, stuff2 + n);
  }
  static_assert(!test1(1), "");
  static_assert(test1(3), "");
  static_assert(!test1(6), "");
  static_assert(test1(9), "");
  static_assert(!test1(100), "");
  static_assert(!test1(101), ""); // expected-error {{constant expression}} expected-note {{in call to 'test1(101)'}}

  // FIXME: We should be able to reject this before it's called
  constexpr void f() {
    char foo[10] = { "z" }; // expected-note {{here}}
    foo[10] = 'x'; // expected-warning {{past the end}} expected-note {{assignment to dereferenced one-past-the-end pointer}}
  }
  constexpr int k = (f(), 0); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace array_resize {
  constexpr int do_stuff(int k1, int k2) {
    int arr[1234] = { 1, 2, 3, 4 };
    arr[k1] = 5; // expected-note {{past-the-end}} expected-note {{cannot refer to element 1235}} expected-note {{cannot refer to element -1}}
    return arr[k2];
  }
  static_assert(do_stuff(1, 2) == 3, "");
  static_assert(do_stuff(0, 0) == 5, "");
  static_assert(do_stuff(1233, 1233) == 5, "");
  static_assert(do_stuff(1233, 0) == 1, "");
  static_assert(do_stuff(1234, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(do_stuff(1235, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(do_stuff(-1, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace potential_const_expr {
  constexpr void set(int &n) { n = 1; }
  constexpr int div_zero_1() { int z = 0; set(z); return 100 / z; } // no error
  constexpr int div_zero_2() { // expected-error {{never produces a constant expression}}
    int z = 0;
    return 100 / (set(z), 0); // expected-note {{division by zero}}
  }
}

namespace subobject {
  union A { constexpr A() : y(5) {} int x, y; };
  struct B { A a; };
  struct C : B {};
  union D { constexpr D() : c() {} constexpr D(int n) : n(n) {} C c; int n; };
  constexpr void f(D &d) {
    d.c.a.y = 3;
    // expected-note@-1 {{cannot modify an object that is visible outside}}
    // expected-note@-2 {{assignment to member 'c' of union with active member 'n'}}
  }
  constexpr bool check(D &d) { return d.c.a.y == 3; }

  constexpr bool g() { D d; f(d); return d.c.a.y == 3; }
  static_assert(g(), "");

  D d;
  constexpr bool h() { f(d); return check(d); } // expected-note {{in call}}
  static_assert(h(), ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr bool i() { D d(0); f(d); return check(d); } // expected-note {{in call}}
  static_assert(i(), ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr bool j() { D d; d.c.a.x = 3; return check(d); } // expected-note {{assignment to member 'x' of union with active member 'y'}}
  static_assert(j(), ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace lifetime {
  constexpr int &&id(int &&n) { return static_cast<int&&>(n); }
  constexpr int &&dead() { return id(0); } // expected-note {{temporary created here}}
  constexpr int bad() { int &&n = dead(); n = 1; return n; } // expected-note {{assignment to temporary whose lifetime has ended}}
  static_assert(bad(), ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace const_modify {
  constexpr int modify(int &n) { return n = 1; } // expected-note {{modification of object of const-qualified type 'const int'}}
  constexpr int test1() { int k = 0; return modify(k); }
  constexpr int test2() { const int k = 0; return modify(const_cast<int&>(k)); } // expected-note {{in call}}
  static_assert(test1() == 1, "");
  static_assert(test2() == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace null {
  constexpr int test(int *p) {
    return *p = 123; // expected-note {{assignment to dereferenced null pointer}}
  }
  static_assert(test(0), ""); // expected-error {{constant expression}} expected-note {{in call}}
}
