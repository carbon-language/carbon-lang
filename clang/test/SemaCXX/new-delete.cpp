// RUN: clang-cc -fsyntax-only -verify %s

#include <stddef.h>

struct S // expected-note {{candidate}}
{
  S(int, int, double); // expected-note {{candidate}}
  S(double, int); // expected-note 2 {{candidate}}
  S(float, int); // expected-note 2 {{candidate}}
};
struct T; // expected-note{{forward declaration of 'struct T'}}
struct U
{
  // A special new, to verify that the global version isn't used.
  void* operator new(size_t, S*); // expected-note {{candidate}}
};
struct V : U
{
};

void* operator new(size_t); // expected-note 2 {{candidate}}
void* operator new(size_t, int*); // expected-note 3 {{candidate}}
void* operator new(size_t, float*); // expected-note 3 {{candidate}}
void* operator new(size_t, S); // expected-note 2 {{candidate}}

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
  ps = new (S[3])(1, 2, 3.4);
  typedef int ia4[4];
  ia4 *pai = new (int[3][4]);
  pi = ::new int;
  U *pu = new (ps) U;
  V *pv = new (ps) V;
  
  pi = new (S(1.0f, 2)) int;
  
  (void)new int[true];
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
  (void)new (int[i]); // expected-error {{when type is in parentheses}}
  (void)new int(*(S*)0); // expected-error {{incompatible type initializing}}
  (void)new int(1, 2); // expected-error {{initializer of a builtin type can only take one argument}}
  (void)new S(1); // expected-error {{no matching constructor}}
  (void)new S(1, 1); // expected-error {{call to constructor of 'S' is ambiguous}}
  (void)new const int; // expected-error {{must provide an initializer}}
  (void)new float*(ip); // expected-error {{incompatible type initializing 'int *', expected 'float *'}}
  // Undefined, but clang should reject it directly.
  (void)new int[-1]; // expected-error {{array size is negative}}
  (void)new int[*(S*)0]; // expected-error {{array size expression must have integral or enumerated type, not 'struct S'}}
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
  delete (void*)0; // expected-error {{cannot delete expression}}
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
  delete x2; // expected-error{{ambiguous conversion of delete expression of type 'struct X2' to a pointer}}
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
  static int operator new(signed char) throw(); // expected-error {{'operator new' takes type size_t}} \
						  // expected-error {{operator new' must return type 'void *'}}
  static int operator new[] (signed char) throw(); // expected-error {{'operator new[]' takes type size_t}} \
						     // expected-error {{operator new[]' must return type 'void *'}}
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
  void* operator new(T size, int); // expected-error {{'operator new' takes type size_t}}
};

TBase<int> t1; // expected-note {{in instantiation of template class 'struct TBase<int>' requested here}}

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
  static void operator delete(void*, int); // expected-note {{'operator delete' declared here}}
  static void operator delete(void*, float); // expected-note {{'operator delete' declared here}}
};

void f(X9 *x9) {
  delete x9; // expected-error {{no suitable member 'operator delete' in 'X9'}}
}
