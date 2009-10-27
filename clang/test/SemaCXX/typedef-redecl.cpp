// RUN: clang-cc -fsyntax-only -verify -fms-extensions=0 %s 
typedef int INT;
typedef INT REALLY_INT; // expected-note {{previous definition is here}}
typedef REALLY_INT REALLY_REALLY_INT;
typedef REALLY_INT BOB;
typedef float REALLY_INT; // expected-error{{typedef redefinition with different types ('float' vs 'INT' (aka 'int'))}}

struct X {
  typedef int result_type; // expected-note {{previous definition is here}}
  typedef INT result_type; // expected-error {{redefinition of 'result_type'}}
};

struct Y; // expected-note{{previous definition is here}}
typedef int Y;  // expected-error{{typedef redefinition with different types ('int' vs 'struct Y')}}

typedef int Y2; // expected-note{{previous definition is here}}
struct Y2; // expected-error{{definition of type 'struct Y2' conflicts with typedef of the same name}}

void f(); // expected-note{{previous definition is here}}
typedef int f; // expected-error{{redefinition of 'f' as different kind of symbol}}

typedef int f2; // expected-note{{previous definition is here}}
void f2(); // expected-error{{redefinition of 'f2' as different kind of symbol}}

typedef struct s s; 
typedef int I; 
typedef int I; 
typedef I I; 

struct s { };

