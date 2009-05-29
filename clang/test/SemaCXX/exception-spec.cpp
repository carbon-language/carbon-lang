// RUN: clang-cc -fsyntax-only -verify %s

// Straight from the standard:
// Plain function with spec
void f() throw(int);
// Pointer to function with spec
void (*fp)() throw (int);
// Function taking reference to function with spec
void g(void pfa() throw(int));
// Typedef for pointer to function with spec
typedef int (*pf)() throw(int); // xpected-error spec-on-typedef

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
