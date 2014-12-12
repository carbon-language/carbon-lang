// RUN: %clang_cc1 -fsyntax-only %s -verify

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0
#define _HAVE_X 0
#define X__Y

#undef __cplusplus
#undef _HAVE_X
#undef X__Y

#pragma clang diagnostic warning "-Wreserved-id-macro"

#define switch if  // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0 // expected-warning {{macro name is a reserved identifier}}
#define _HAVE_X 0  // expected-warning {{macro name is a reserved identifier}}
#define X__Y

#undef __cplusplus // expected-warning {{macro name is a reserved identifier}}
#undef _HAVE_X     // expected-warning {{macro name is a reserved identifier}}
#undef X__Y

int x;
