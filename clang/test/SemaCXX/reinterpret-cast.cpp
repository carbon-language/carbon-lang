// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s

#include <stdint.h>

enum test { testval = 1 };
struct structure { int m; };
typedef void (*fnptr)();

// Test the conversion to self.
void self_conversion()
{
  // T*->T* is allowed, T->T in general not.
  int i = 0;
  (void)reinterpret_cast<int>(i); // expected-error {{reinterpret_cast from 'int' to 'int' is not allowed}}
  structure s;
  (void)reinterpret_cast<structure>(s); // expected-error {{reinterpret_cast from 'structure' to 'structure' is not allowed}}
  int *pi = 0;
  (void)reinterpret_cast<int*>(pi);
}

// Test conversion between pointer and integral types, as in /3 and /4.
void integral_conversion()
{
  void *vp = reinterpret_cast<void*>(testval);
  intptr_t i = reinterpret_cast<intptr_t>(vp);
  (void)reinterpret_cast<float*>(i);
  fnptr fnp = reinterpret_cast<fnptr>(i);
  (void)reinterpret_cast<char>(fnp); // expected-error {{cast from pointer to smaller type 'char' loses information}}
  (void)reinterpret_cast<intptr_t>(fnp);
}

void pointer_conversion()
{
  int *p1 = 0;
  float *p2 = reinterpret_cast<float*>(p1);
  structure *p3 = reinterpret_cast<structure*>(p2);
  typedef int **ppint;
  ppint *deep = reinterpret_cast<ppint*>(p3);
  (void)reinterpret_cast<fnptr*>(deep);
}

void constness()
{
  int ***const ipppc = 0;
  // Valid: T1* -> T2 const*
  int const *icp = reinterpret_cast<int const*>(ipppc);
  // Invalid: T1 const* -> T2*
  (void)reinterpret_cast<int*>(icp); // expected-error {{reinterpret_cast from 'const int *' to 'int *' casts away qualifiers}}
  // Invalid: T1*** -> T2 const* const**
  int const *const **icpcpp = reinterpret_cast<int const* const**>(ipppc); // expected-error {{reinterpret_cast from 'int ***' to 'const int *const **' casts away qualifiers}}
  // Valid: T1* -> T2*
  int *ip = reinterpret_cast<int*>(icpcpp);
  // Valid: T* -> T const*
  (void)reinterpret_cast<int const*>(ip);
  // Valid: T*** -> T2 const* const* const*
  (void)reinterpret_cast<int const* const* const*>(ipppc);
}

void fnptrs()
{
  typedef int (*fnptr2)(int);
  fnptr fp = 0;
  (void)reinterpret_cast<fnptr2>(fp);
  void *vp = reinterpret_cast<void*>(fp);
  (void)reinterpret_cast<fnptr>(vp);
}

void refs()
{
  long l = 0;
  char &c = reinterpret_cast<char&>(l);
  // Bad: from rvalue
  (void)reinterpret_cast<int&>(&c); // expected-error {{reinterpret_cast from rvalue to reference type 'int &'}}
}

void memptrs()
{
  const int structure::*psi = 0;
  (void)reinterpret_cast<const float structure::*>(psi);
  (void)reinterpret_cast<int structure::*>(psi); // expected-error {{reinterpret_cast from 'const int structure::*' to 'int structure::*' casts away qualifiers}}

  void (structure::*psf)() = 0;
  (void)reinterpret_cast<int (structure::*)()>(psf);

  (void)reinterpret_cast<void (structure::*)()>(psi); // expected-error {{reinterpret_cast from 'const int structure::*' to 'void (structure::*)()' is not allowed}}
  (void)reinterpret_cast<int structure::*>(psf); // expected-error {{reinterpret_cast from 'void (structure::*)()' to 'int structure::*' is not allowed}}

  // Cannot cast from integers to member pointers, not even the null pointer
  // literal.
  (void)reinterpret_cast<void (structure::*)()>(0); // expected-error {{reinterpret_cast from 'int' to 'void (structure::*)()' is not allowed}}
  (void)reinterpret_cast<int structure::*>(0); // expected-error {{reinterpret_cast from 'int' to 'int structure::*' is not allowed}}
}

namespace PR5545 {
// PR5545
class A;
class B;
void (A::*a)();
void (B::*b)() = reinterpret_cast<void (B::*)()>(a);
}

// <rdar://problem/8018292>
void const_arrays() {
  typedef char STRING[10];
  const STRING *s;
  const char *c;

  (void)reinterpret_cast<char *>(s); // expected-error {{reinterpret_cast from 'const STRING *' (aka 'char const (*)[10]') to 'char *' casts away qualifiers}}
  (void)reinterpret_cast<const STRING *>(c);
}
