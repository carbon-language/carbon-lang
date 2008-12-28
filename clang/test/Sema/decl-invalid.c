// RUN: clang %s -fsyntax-only -verify

// See Sema::ParsedFreeStandingDeclSpec about the double diagnostic
typedef union <anonymous> __mbstate_t;  // expected-error {{declaration of anonymous union must be a definition}} expected-error {{declaration does not declare anything}}


// PR2017
void x(); 
int a() {
  int r[x()];  // expected-error {{size of array has non-integer type 'void'}}
}

int; // expected-error {{declaration does not declare anything}}
typedef int; // expected-error {{declaration does not declare anything}}
const int; // expected-error {{declaration does not declare anything}}
struct; // expected-error {{declaration of anonymous struct must be a definition}} // expected-error {{declaration does not declare anything}}
typedef int I;
I; // expected-error {{declaration does not declare anything}}
