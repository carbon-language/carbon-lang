// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

__attribute((regparm(2))) int x(void);
__attribute((regparm(1.0))) int x(void); // expected-error{{'regparm' attribute requires integer constant}}
__attribute((regparm(-1))) int x(void); // expected-error{{'regparm' parameter must be between 0 and 3 inclusive}}
__attribute((regparm(5))) int x(void); // expected-error{{'regparm' parameter must be between 0 and 3 inclusive}}
__attribute((regparm(5,3))) int x(void); // expected-error{{attribute requires 1 argument(s)}}
