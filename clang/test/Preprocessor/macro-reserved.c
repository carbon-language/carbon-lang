// RUN: %clang_cc1 -fsyntax-only %s -verify

#pragma clang diagnostic warning "-Wkeyword-macro"

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0
#define _HAVE_X 0
#define X__Y

#undef __cplusplus
#undef _HAVE_X
#undef X__Y

#define switch if  // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0
#define _HAVE_X 0
#define X__Y

#undef __cplusplus
#undef _HAVE_X
#undef X__Y

int x;
