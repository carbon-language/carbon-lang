// RUN: %clang_cc1 -fsyntax-only -Wall -Wuninitialized -verify %s

int foo(int x);
int bar(int* x);
int boo(int& x);
int far(const int& x);

// Test self-references within initializers which are guaranteed to be
// uninitialized.
int a = a; // no-warning: used to signal intended lack of initialization.
int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
int c = (c + c); // expected-warning 2 {{variable 'c' is uninitialized when used within its own initialization}}
void test() {
  int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
}
int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

// Thes don't warn as they don't require the value.
int g = sizeof(g);
void* ptr = &ptr;
int h = bar(&h);
int i = boo(i);
int j = far(j);
int k = __alignof__(k);


// Test self-references with record types.
class A {
  // Non-POD class.
  public:
    enum count { ONE, TWO, THREE };
    int num;
    static int count;
    int get() const { return num; }
    int get2() { return num; }
    void set(int x) { num = x; }
    static int zero() { return 0; }

    A() {}
    A(A const &a) {}
    A(int x) {}
    A(int *x) {}
    A(A *a) {}
    ~A();
};

A getA() { return A(); }
A getA(int x) { return A(); }
A getA(A* a) { return A(); }

void setupA() {
  A a1;
  a1.set(a1.get());
  A a2(a1.get());
  A a3(a1);
  A a4(&a4);
  A a5(a5.zero());
  A a6(a6.ONE);
  A a7 = getA();
  A a8 = getA(a8.TWO);
  A a9 = getA(&a9);
  A a10(a10.count);

  A a11(a11);  // expected-warning {{variable 'a11' is uninitialized when used within its own initialization}}
  A a12(a12.get());  // expected-warning {{variable 'a12' is uninitialized when used within its own initialization}}
  A a13(a13.num);  // expected-warning {{variable 'a13' is uninitialized when used within its own initialization}}
  A a14 = A(a14);  // expected-warning {{variable 'a14' is uninitialized when used within its own initialization}}
  A a15 = getA(a15.num);  // expected-warning {{variable 'a15' is uninitialized when used within its own initialization}}
  A a16(&a16.num);  // expected-warning {{variable 'a16' is uninitialized when used within its own initialization}}
  A a17(a17.get2());  // expected-warning {{variable 'a17' is uninitialized when used within its own initialization}}
}

struct B {
  // POD struct.
  int x;
  int *y;
};

B getB() { return B(); };
B getB(int x) { return B(); };
B getB(int *x) { return B(); };
B getB(B *b) { return B(); };

void setupB() {
  B b1;
  B b2(b1);
  B b3 = { 5, &b3.x };
  B b4 = getB();
  B b5 = getB(&b5);
  B b6 = getB(&b6.x);

  // Silence unused warning
  (void) b2;
  (void) b4;

  B b7(b7);  // expected-warning {{variable 'b7' is uninitialized when used within its own initialization}}
  B b8 = getB(b8.x);  // expected-warning {{variable 'b8' is uninitialized when used within its own initialization}}
  B b9 = getB(b9.y);  // expected-warning {{variable 'b9' is uninitialized when used within its own initialization}}
}

// Also test similar constructs in a field's initializer.
struct S {
  int x;
  void *ptr;

  S(bool (*)[1]) : x(x) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[2]) : x(x + 1) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[3]) : x(x + x) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[4]) : x(static_cast<long>(x) + 1) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[5]) : x(foo(x)) {} // FIXME: This should warn!

  // These don't actually require the value of x and so shouldn't warn.
  S(char (*)[1]) : x(sizeof(x)) {} // rdar://8610363
  S(char (*)[2]) : ptr(&ptr) {}
  S(char (*)[3]) : x(__alignof__(x)) {}
  S(char (*)[4]) : x(bar(&x)) {}
  S(char (*)[5]) : x(boo(x)) {}
  S(char (*)[6]) : x(far(x)) {}
};

struct C { char a[100], *e; } car = { .e = car.a };

// <rdar://problem/10398199>
namespace rdar10398199 {
  class FooBase { protected: ~FooBase() {} };
  class Foo : public FooBase {
  public:
    operator int&() const;
  };
  void stuff();
  template <typename T> class FooImpl : public Foo {
    T val;
  public:
    FooImpl(const T &x) : val(x) {}
    ~FooImpl() { stuff(); }
  };

  template <typename T> FooImpl<T> makeFoo(const T& x) {
    return FooImpl<T>(x);
  }

  void test() {
    const Foo &x = makeFoo(42);
    const int&y = makeFoo(42u);
    (void)x;
    (void)y;
  };
}

// PR 12325 - this was a false uninitialized value warning due to
// a broken CFG.
int pr12325(int params) {
  int x = ({
    while (false)
      ;
    int _v = params;
    if (false)
      ;
    _v; // no-warning
  });
  return x;
}

