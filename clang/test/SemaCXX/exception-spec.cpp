// RUN: clang-cc -fsyntax-only -verify -fms-extensions %s

// Straight from the standard:
// Plain function with spec
void f() throw(int);
// Pointer to function with spec
void (*fp)() throw (int);
// Function taking reference to function with spec
void g(void pfa() throw(int));
// Typedef for pointer to function with spec
typedef int (*pf)() throw(int); // expected-error {{specifications are not allowed in typedefs}}

// Some more:
// Function returning function with spec
void (*h())() throw(int);
// Ultimate parser thrill: function with spec returning function with spec and
// taking pointer to function with spec.
// The actual function throws int, the return type double, the argument float.
void (*i() throw(int))(void (*)() throw(float)) throw(double);
// Pointer to pointer to function taking function with spec
void (**k)(void pfa() throw(int)); // no-error
// Pointer to pointer to function with spec
void (**j)() throw(int); // expected-error {{not allowed beyond a single}}
// Pointer to function returning pointer to pointer to function with spec
void (**(*h())())() throw(int); // expected-error {{not allowed beyond a single}}

struct Incomplete;

// Exception spec must not have incomplete types, or pointers to them, except
// void.
void ic1() throw(void); // expected-error {{incomplete type 'void' is not allowed in exception specification}}
void ic2() throw(Incomplete); // expected-error {{incomplete type 'struct Incomplete' is not allowed in exception specification}}
void ic3() throw(void*);
void ic4() throw(Incomplete*); // expected-error {{pointer to incomplete type 'struct Incomplete' is not allowed in exception specification}}
void ic5() throw(Incomplete&); // expected-error {{reference to incomplete type 'struct Incomplete' is not allowed in exception specification}}

// Redeclarations
typedef int INT;
void r1() throw(int);
void r1() throw(int);

void r2() throw(int);
void r2() throw(INT);

// throw-any spec and no spec at all are semantically equivalent
void r3();
void r3() throw(...);

void r4() throw(int, float);
void r4() throw(float, int);

void r5() throw(int); // expected-note {{previous declaration}}
void r5(); // expected-error {{exception specification in declaration does not match}}

void r6() throw(...); // expected-note {{previous declaration}}
void r6() throw(int); // expected-error {{exception specification in declaration does not match}}

void r7() throw(int); // expected-note {{previous declaration}}
void r7() throw(float); // expected-error {{exception specification in declaration does not match}}

// Top-level const doesn't matter.
void r8() throw(int);
void r8() throw(const int);

struct A
{
};

struct B1 : A
{
};

struct B2 : A
{
};

struct D : B1, B2
{
};

struct P : private A
{
};

struct Base
{
  virtual void f1() throw();
  virtual void f2();
  virtual void f3() throw(...);
  virtual void f4() throw(int, float);

  virtual void f5() throw(int, float);
  virtual void f6() throw(A);
  virtual void f7() throw(A, int, float);
  virtual void f8();

  virtual void g1() throw(); // expected-note {{overridden virtual function is here}}
  virtual void g2() throw(int); // expected-note {{overridden virtual function is here}}
  virtual void g3() throw(A); // expected-note {{overridden virtual function is here}}
  virtual void g4() throw(B1); // expected-note {{overridden virtual function is here}}
  virtual void g5() throw(A); // expected-note {{overridden virtual function is here}}
};
struct Derived : Base
{
  virtual void f1() throw();
  virtual void f2() throw(...);
  virtual void f3();
  virtual void f4() throw(float, int);

  virtual void f5() throw(float);
  virtual void f6() throw(B1);
  virtual void f7() throw(B1, B2, int);
  virtual void f8() throw(B2, B2, int, float, char, double, bool);

  virtual void g1() throw(int); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g2(); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g3() throw(D); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g4() throw(A); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g5() throw(P); // expected-error {{exception specification of overriding function is more lax}}
};

// Some functions to play with below.
void s1() throw();
void s2() throw(int);
void s3() throw(A);
void s4() throw(B1);
void s5() throw(D);
void s6();
void s7() throw(int, float);
void (*s8())() throw(B1); // s8 returns a pointer to function with spec
void s9(void (*)() throw(B1)); // s9 takes pointer to function with spec

void fnptrs()
{
  // Assignment and initialization of function pointers.
  void (*t1)() throw() = &s1;    // valid
  t1 = &s2;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  t1 = &s3;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  void (&t2)() throw() = s2;     // expected-error {{not superset}}
  void (*t3)() throw(int) = &s2; // valid
  void (*t4)() throw(A) = &s1;   // valid
  t4 = &s3;                      // valid
  t4 = &s4;                      // valid
  t4 = &s5;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  void (*t5)() = &s1;            // valid
  t5 = &s2;                      // valid
  t5 = &s6;                      // valid
  t5 = &s7;                      // valid
  t1 = t3;                       // expected-error {{not superset}} expected-error {{incompatible type}}
  t3 = t1;                       // valid
  void (*t6)() throw(B1);
  t6 = t4;                       // expected-error {{not superset}} expected-error {{incompatible type}}
  t4 = t6;                       // valid
  t5 = t1;                       // valid
  t1 = t5;                       // expected-error {{not superset}} expected-error {{incompatible type}}

  // return types and arguments must match exactly, no inheritance allowed
  void (*(*t7)())() throw(B1) = &s8;       // valid
  void (*(*t8)())() throw(A) = &s8;        // expected-error {{return types differ}} expected-error {{incompatible type}}
  void (*(*t9)())() throw(D) = &s8;        // expected-error {{return types differ}} expected-error {{incompatible type}}
  void (*t10)(void (*)() throw(B1)) = &s9; // valid   expected-warning{{disambiguated}}
  void (*t11)(void (*)() throw(A)) = &s9;  // expected-error {{argument types differ}} expected-error {{incompatible type}} expected-warning{{disambiguated}}
  void (*t12)(void (*)() throw(D)) = &s9;  // expected-error {{argument types differ}} expected-error {{incompatible type}} expected-warning{{disambiguated}}
}
