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
    while (begin != end && begin != --end)
      swap(*begin++, *end);
  }
  template<typename Iterator1, typename Iterator2>
  constexpr bool equal(Iterator1 a, Iterator1 ae, Iterator2 b, Iterator2 be) {
    while (a != ae && b != be)
      if (*a++ != *b++)
        return false;
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
  int n; // expected-note {{declared here}}
  constexpr int ref() { // expected-error {{never produces a constant expression}}
    int &r = n;
    return r; // expected-note {{read of non-const variable 'n'}}
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

namespace incdec {
  template<typename T> constexpr T &ref(T &&r) { return r; }
  template<typename T> constexpr T postinc(T &&r) { return (r++, r); }
  template<typename T> constexpr T postdec(T &&r) { return (r--, r); }

  static_assert(++ref(0) == 1, "");
  static_assert(ref(0)++ == 0, "");
  static_assert(postinc(0) == 1, "");
  static_assert(--ref(0) == -1, "");
  static_assert(ref(0)-- == 0, "");
  static_assert(postdec(0) == -1, "");

  constexpr int overflow_int_inc_1 = ref(0x7fffffff)++; // expected-error {{constant}} expected-note {{2147483648}}
  constexpr int overflow_int_inc_1_ok = ref(0x7ffffffe)++;
  constexpr int overflow_int_inc_2 = ++ref(0x7fffffff); // expected-error {{constant}} expected-note {{2147483648}}
  constexpr int overflow_int_inc_2_ok = ++ref(0x7ffffffe);

  // inc/dec on short can't overflow because we promote to int first
  static_assert(++ref<short>(0x7fff) == (int)0xffff8000u, "");
  static_assert(--ref<short>(0x8000) == 0x7fff, "");

  // inc on bool sets to true
  static_assert(++ref(false), ""); // expected-warning {{deprecated}}
  static_assert(++ref(true), ""); // expected-warning {{deprecated}}

  int arr[10];
  static_assert(++ref(&arr[0]) == &arr[1], "");
  static_assert(++ref(&arr[9]) == &arr[10], "");
  static_assert(++ref(&arr[10]) == &arr[11], ""); // expected-error {{constant}} expected-note {{cannot refer to element 11}}
  static_assert(ref(&arr[0])++ == &arr[0], "");
  static_assert(ref(&arr[10])++ == &arr[10], ""); // expected-error {{constant}} expected-note {{cannot refer to element 11}}
  static_assert(postinc(&arr[0]) == &arr[1], "");
  static_assert(--ref(&arr[10]) == &arr[9], "");
  static_assert(--ref(&arr[1]) == &arr[0], "");
  static_assert(--ref(&arr[0]) != &arr[0], ""); // expected-error {{constant}} expected-note {{cannot refer to element -1}}
  static_assert(ref(&arr[1])-- == &arr[1], "");
  static_assert(ref(&arr[0])-- == &arr[0], ""); // expected-error {{constant}} expected-note {{cannot refer to element -1}}
  static_assert(postdec(&arr[1]) == &arr[0], "");

  int x;
  static_assert(++ref(&x) == &x + 1, "");

  static_assert(++ref(0.0) == 1.0, "");
  static_assert(ref(0.0)++ == 0.0, "");
  static_assert(postinc(0.0) == 1.0, "");
  static_assert(--ref(0.0) == -1.0, "");
  static_assert(ref(0.0)-- == 0.0, "");
  static_assert(postdec(0.0) == -1.0, "");

  static_assert(++ref(1e100) == 1e100, "");
  static_assert(--ref(1e100) == 1e100, "");

  union U {
    int a, b;
  };
  constexpr int f(U u) {
    return ++u.b; // expected-note {{increment of member 'b' of union with active member 'a'}}
  }
  constexpr int wrong_member = f({0}); // expected-error {{constant}} expected-note {{in call to 'f({.a = 0})'}}
  constexpr int vol = --ref<volatile int>(0); // expected-error {{constant}} expected-note {{decrement of volatile-qualified}}

  constexpr int incr(int k) {
    int x = k;
    if (x++ == 100)
      return x;
    return incr(x);
  }
  static_assert(incr(0) == 101, "");
}

namespace loops {
  constexpr int fib_loop(int a) {
    int f_k = 0, f_k_plus_one = 1;
    for (int k = 1; k != a; ++k) {
      int f_k_plus_two = f_k + f_k_plus_one;
      f_k = f_k_plus_one;
      f_k_plus_one = f_k_plus_two;
    }
    return f_k_plus_one;
  }
  static_assert(fib_loop(46) == 1836311903, "");

  constexpr bool breaks_work() {
    int a = 0;
    for (int n = 0; n != 100; ++n) {
      ++a;
      if (a == 5) continue;
      if ((a % 5) == 0) break;
    }

    int b = 0;
    while (b != 17) {
      ++b;
      if (b == 6) continue;
      if ((b % 6) == 0) break;
    }

    int c = 0;
    do {
      ++c;
      if (c == 7) continue;
      if ((c % 7) == 0) break;
    } while (c != 21);

    return a == 10 && b == 12 & c == 14;
  }
  static_assert(breaks_work(), "");

  void not_constexpr();
  constexpr bool no_cont_after_break() {
    for (;;) {
      break;
      not_constexpr();
    }
    while (true) {
      break;
      not_constexpr();
    }
    do {
      break;
      not_constexpr();
    } while (true);
    return true;
  }
  static_assert(no_cont_after_break(), "");

  constexpr bool cond() {
    for (int a = 1; bool b = a != 3; ++a) {
      if (!b)
        return false;
    }
    while (bool b = true) {
      b = false;
      break;
    }
    return true;
  }
  static_assert(cond(), "");

  constexpr int range_for() {
    int arr[] = { 1, 2, 3, 4, 5 };
    int sum = 0;
    for (int x : arr)
      sum = sum + x;
    return sum;
  }
  static_assert(range_for() == 15, "");

  template<int...N> struct ints {};
  template<typename A, typename B> struct join_ints;
  template<int...As, int...Bs> struct join_ints<ints<As...>, ints<Bs...>> {
    using type = ints<As..., sizeof...(As) + Bs...>;
  };
  template<unsigned N> struct make_ints {
    using type = typename join_ints<typename make_ints<N/2>::type, typename make_ints<(N+1)/2>::type>::type;
  };
  template<> struct make_ints<0> { using type = ints<>; };
  template<> struct make_ints<1> { using type = ints<0>; };

  struct ignore { template<typename ...Ts> constexpr ignore(Ts &&...) {} };

  template<typename T, unsigned N> struct array {
    constexpr array() : arr{} {}
    template<typename ...X>
    constexpr array(X ...x) : arr{} {
      init(typename make_ints<sizeof...(X)>::type{}, x...);
    }
    template<int ...I, typename ...X> constexpr void init(ints<I...>, X ...x) {
      ignore{arr[I] = x ...};
    }
    T arr[N];
    struct iterator {
      T *p;
      constexpr explicit iterator(T *p) : p(p) {}
      constexpr bool operator!=(iterator o) { return p != o.p; }
      constexpr iterator &operator++() { ++p; return *this; }
      constexpr T &operator*() { return *p; }
    };
    constexpr iterator begin() { return iterator(arr); }
    constexpr iterator end() { return iterator(arr + N); }
  };

  constexpr int range_for_2() {
    array<int, 5> arr { 1, 2, 3, 4, 5 };
    int sum = 0;
    for (int k : arr) {
      sum = sum + k;
      if (sum > 8) break;
    }
    return sum;
  }
  static_assert(range_for_2() == 10, "");
}
