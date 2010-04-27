// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

__attribute((regparm(2))) int a(void);
__attribute((regparm(1.0))) int b(void); // expected-error{{'regparm' attribute requires integer constant}}
__attribute((regparm(-1))) int c(void); // expected-error{{'regparm' parameter must be between 0 and 3 inclusive}}
__attribute((regparm(5))) int d(void); // expected-error{{'regparm' parameter must be between 0 and 3 inclusive}}
__attribute((regparm(5,3))) int e(void); // expected-error{{attribute requires 1 argument(s)}}
int f(void);
__attribute((regparm(0))) int f(void);
__attribute((regparm(1))) int g(void); // expected-note{{previous declaration is here}}
__attribute((regparm(2))) int g(void); // expected-error{{conflicting types for 'g'}}
