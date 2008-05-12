// RUN: clang %s -verify -fsyntax-only

const int a [1] = {1};
extern const int a[];

extern const int b[];
const int b [1] = {1};

extern const int c[] = {1}; // expected-warning{{'extern' variable has an initializer}}
const int c[];

int i1 = 1; // expected-error{{previous definition is here}}
int i1 = 2; // expected-error{{redefinition of 'i1'}} // expected-error{{previous definition is here}}
// FIXME: the following should not be an error (see related FIXME in Sema::MergeVarDecl). 
int i1; // expected-error{{redefinition of 'i1'}}
int i1;
extern int i1; // expected-error{{previous definition is here}}
static int i1; // expected-error{{static declaration of 'i1' follows non-static declaration}} expected-error{{previous definition is here}}
int i1 = 3; // expected-error{{non-static declaration of 'i1' follows static declaration}}

__private_extern__ int pExtern;
int pExtern = 0;

void func() {
  extern int i1; // expected-error{{previous definition is here}}
  static int i1; // expected-error{{static declaration of 'i1' follows non-static declaration}}
}
