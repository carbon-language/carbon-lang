// RUN: clang-cc -fsyntax-only -verify %s

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
