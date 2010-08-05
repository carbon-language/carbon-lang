// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stddef.h>

struct S // expected-note {{candidate}}
{
  S(int, int, double); // expected-note {{candidate}}
  S(double, int); // expected-note 2 {{candidate}}
  S(float, int); // expected-note 2 {{candidate}}
};
struct T; // expected-note{{forward declaration of 'T'}}
struct U
{
  // A special new, to verify that the global version isn't used.
  void* operator new(size_t, S*); // expected-note {{candidate}}
};
struct V : U
{
};

// PR5823
void* operator new(const size_t); // expected-note 2 {{candidate}}
void* operator new(size_t, int*); // expected-note 3 {{candidate}}
void* operator new(size_t, float*); // expected-note 3 {{candidate}}
void* operator new(size_t, S); // expected-note 2 {{candidate}}

struct foo { };

void good_news()
{
  int *pi = new int;
  float *pf = new (pi) float();
  pi = new int(1);
  pi = new int('c');
  const int *pci = new const int();
  S *ps = new S(1, 2, 3.4);
  ps = new (pf) (S)(1, 2, 3.4);
  S *(*paps)[2] = new S*[*pi][2];
  typedef int ia4[4];
  ia4 *pai = new (int[3][4]);
  pi = ::new int;
  U *pu = new (ps) U;
  V *pv = new (ps) V;
  
  pi = new (S(1.0f, 2)) int;
  
  (void)new int[true];

  // PR7147
  typedef int a[2];
  foo* f1 = new foo;
  foo* f2 = new foo[2];
  typedef foo x[2];
  typedef foo y[2][2];
  x* f3 = new y;
}

struct abstract {
  virtual ~abstract() = 0;
};

void bad_news(int *ip)
{
  int i = 1;
  (void)new; // expected-error {{missing type specifier}}
  (void)new 4; // expected-error {{missing type specifier}}
  (void)new () int; // expected-error {{expected expression}}
  (void)new int[1.1]; // expected-error {{array size expression must have integral or enumerated type, not 'double'}}
  (void)new int[1][i]; // expected-error {{only the first dimension}}
  (void)new (int[1][i]); // expected-error {{only the first dimension}}
  (void)new (int[i]); // expected-warning {{when type is in parentheses}}
  (void)new int(*(S*)0); // expected-error {{no viable conversion from 'S' to 'int'}}
  (void)new int(1, 2); // expected-error {{excess elements in scalar initializer}}
  (void)new S(1); // expected-error {{no matching constructor}}
  (void)new S(1, 1); // expected-error {{call to constructor of 'S' is ambiguous}}
  (void)new const int; // expected-error {{default initialization of an object of const type 'int const'}}
  (void)new float*(ip); // expected-error {{cannot initialize a new value of type 'float *' with an lvalue of type 'int *'}}
  // Undefined, but clang should reject it directly.
  (void)new int[-1]; // expected-error {{array size is negative}}
  (void)new int[*(S*)0]; // expected-error {{array size expression must have integral or enumerated type, not 'S'}}
  (void)::S::new int; // expected-error {{expected unqualified-id}}
  (void)new (0, 0) int; // expected-error {{no matching function for call to 'operator new'}}
  (void)new (0L) int; // expected-error {{call to 'operator new' is ambiguous}}
  // This must fail, because the member version shouldn't be found.
  (void)::new ((S*)0) U; // expected-error {{no matching function for call to 'operator new'}}
  // This must fail, because any member version hides all global versions.
  (void)new U; // expected-error {{no matching function for call to 'operator new'}}
  (void)new (int[]); // expected-error {{array size must be specified in new expressions}}
  (void)new int&; // expected-error {{cannot allocate reference type 'int &' with new}}
  // Some lacking cases due to lack of sema support.
}

void good_deletes()
{
  delete (int*)0;
  delete [](int*)0;
  delete (S*)0;
  ::delete (int*)0;
}

void bad_deletes()
{
  delete 0; // expected-error {{cannot delete expression of type 'int'}}
  delete [0] (int*)0; // expected-error {{expected ']'}} \
                      // expected-note {{to match this '['}}
  delete (void*)0; // expected-warning {{cannot delete expression with pointer-to-'void' type 'void *'}}
  delete (T*)0; // expected-warning {{deleting pointer to incomplete type}}
  ::S::delete (int*)0; // expected-error {{expected unqualified-id}}
}

struct X0 { };

struct X1 {
  operator int*();
  operator float();
};

struct X2 {
  operator int*(); // expected-note {{candidate function}}
  operator float*(); // expected-note {{candidate function}}
};

void test_delete_conv(X0 x0, X1 x1, X2 x2) {
  delete x0; // expected-error{{cannot delete}}
  delete x1;
  delete x2; // expected-error{{ambiguous conversion of delete expression of type 'X2' to a pointer}}
}

// PR4782
class X3 {
public:
  static void operator delete(void * mem, size_t size);
};

class X4 {
public:
  static void release(X3 *x);
  static void operator delete(void * mem, size_t size);
};


void X4::release(X3 *x) {
  delete x;
}

class X5 {
public:
  void Destroy() const { delete this; }
};

class Base {
public:
  static void *operator new(signed char) throw(); // expected-error {{'operator new' takes type size_t}}
  static int operator new[] (size_t) throw(); // expected-error {{operator new[]' must return type 'void *'}}
};

class Tier {};
class Comp : public Tier {};

class Thai : public Base {
public:
  Thai(const Tier *adoptDictionary);
};

void loadEngineFor() {
  const Comp *dict;
  new Thai(dict);
}

template <class T> struct TBase {
  void* operator new(T size, int); // expected-error {{'operator new' cannot take a dependent type as first parameter; use size_t}}
};

TBase<int> t1;

class X6 {
public:
  static void operator delete(void*, int); // expected-note {{member found by ambiguous name lookup}}
};

class X7 {
public:
  static void operator delete(void*, int); // expected-note {{member found by ambiguous name lookup}}
};

class X8 : public X6, public X7 {
};

void f(X8 *x8) {
  delete x8; // expected-error {{member 'operator delete' found in multiple base classes of different types}}
}

class X9 {
public:
  static void operator delete(void*, int); // expected-note {{'operator delete' declared here}}
  static void operator delete(void*, float); // expected-note {{'operator delete' declared here}}
};

void f(X9 *x9) {
  delete x9; // expected-error {{no suitable member 'operator delete' in 'X9'}}
}

struct X10 {
  virtual ~X10();
};

struct X11 : X10 { // expected-error {{no suitable member 'operator delete' in 'X11'}}
  void operator delete(void*, int); // expected-note {{'operator delete' declared here}}
};

void f() {
  X11 x11; // expected-note {{implicit default destructor for 'X11' first required here}}
}

struct X12 {
  void* operator new(size_t, void*);
};

struct X13 : X12 {
  using X12::operator new;
};

static void* f(void* g)
{
    return new (g) X13();
}

class X14 {
public:
  static void operator delete(void*, const size_t);
};

void f(X14 *x14a, X14 *x14b) {
  delete x14a;
}

namespace PR5918 { // Look for template operator new overloads.
  struct S { template<typename T> static void* operator new(size_t, T); };
  void test() {
    (void)new(0) S;
  }
}

namespace Test1 {

void f() {
  (void)new int[10](1, 2); // expected-error {{array 'new' cannot have initialization arguments}}
  
  typedef int T[10];
  (void)new T(1, 2); // expected-error {{array 'new' cannot have initialization arguments}}
}

template<typename T>
void g(unsigned i) {
  (void)new T[1](i); // expected-error {{array 'new' cannot have initialization arguments}}
}

template<typename T>
void h(unsigned i) {
  (void)new T(i); // expected-error {{array 'new' cannot have initialization arguments}}
}
template void h<unsigned>(unsigned);
template void h<unsigned[10]>(unsigned); // expected-note {{in instantiation of function template specialization 'Test1::h<unsigned int [10]>' requested here}}

}

// Don't diagnose access for overload candidates that aren't selected.
namespace PR7436 {
struct S1 {
  void* operator new(size_t);
  void operator delete(void* p);

private:
  void* operator new(size_t, void*); // expected-note {{declared private here}}
  void operator delete(void*, void*);
};
class S2 {
  void* operator new(size_t); // expected-note {{declared private here}}
  void operator delete(void* p); // expected-note {{declared private here}}
};

void test(S1* s1, S2* s2) { 
  delete s1;
  delete s2; // expected-error {{is a private member}}
  (void)new S1();
  (void)new (0L) S1(); // expected-error {{is a private member}}
  (void)new S2(); // expected-error {{is a private member}}
}
}

namespace rdar8018245 {
  struct X0 {
    static const int value = 17;
  };

  const int X0::value;

  struct X1 {
    static int value;
  };

  int X1::value;

  template<typename T>
  int *f() {
    return new (int[T::value]); // expected-warning{{when type is in parentheses, array cannot have dynamic size}}
  }

  template int *f<X0>();
  template int *f<X1>(); // expected-note{{in instantiation of}}

}

// <rdar://problem/8248780>
namespace Instantiate {
  template<typename T> struct X { 
    operator T*();
  };

  void f(X<int> &xi) {
    delete xi;
  }
}

namespace PR7810 {
  struct X {
    // cv is ignored in arguments
    static void operator delete(void *const);
  };
  struct Y {
    // cv is ignored in arguments
    static void operator delete(void *volatile);
  };
}
