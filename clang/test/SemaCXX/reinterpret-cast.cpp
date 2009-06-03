// RUN: clang-cc -fsyntax-only -verify %s

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
  (void)reinterpret_cast<structure>(s); // expected-error {{reinterpret_cast from 'struct structure' to 'struct structure' is not allowed}}
  int *pi = 0;
  (void)reinterpret_cast<int*>(pi);
}

// Test conversion between pointer and integral types, as in /3 and /4.
void integral_conversion()
{
  void *vp = reinterpret_cast<void*>(testval);
  long l = reinterpret_cast<long>(vp);
  (void)reinterpret_cast<float*>(l);
  fnptr fnp = reinterpret_cast<fnptr>(l);
  (void)reinterpret_cast<char>(fnp); // expected-error {{cast from pointer to smaller type 'char' loses information}}
  (void)reinterpret_cast<long>(fnp);
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
  (void)reinterpret_cast<int*>(icp); // expected-error {{reinterpret_cast from 'int const *' to 'int *' casts away constness}}
  // Invalid: T1*** -> T2 const* const**
  int const *const **icpcpp = reinterpret_cast<int const* const**>(ipppc); // expected-error {{reinterpret_cast from 'int ***const' to 'int const *const **' casts away constness}}
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
  (void)reinterpret_cast<int structure::*>(psi); // expected-error {{reinterpret_cast from 'int const struct structure::*' to 'int struct structure::*' casts away constness}}

  void (structure::*psf)() = 0;
  (void)reinterpret_cast<int (structure::*)()>(psf);

  (void)reinterpret_cast<void (structure::*)()>(psi); // expected-error {{reinterpret_cast from 'int const struct structure::*' to 'void (struct structure::*)()' is not allowed}}
  (void)reinterpret_cast<int structure::*>(psf); // expected-error {{reinterpret_cast from 'void (struct structure::*)()' to 'int struct structure::*' is not allowed}}

  // Cannot cast from integers to member pointers, not even the null pointer
  // literal.
  (void)reinterpret_cast<void (structure::*)()>(0); // expected-error {{reinterpret_cast from 'int' to 'void (struct structure::*)()' is not allowed}}
  (void)reinterpret_cast<int structure::*>(0); // expected-error {{reinterpret_cast from 'int' to 'int struct structure::*' is not allowed}}
}
