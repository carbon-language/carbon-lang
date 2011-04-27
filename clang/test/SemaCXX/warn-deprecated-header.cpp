// RUN: %clang_cc1 -fsyntax-only -fdeprecated-macro -verify %s
// RUN: %clang_cc1 -fsyntax-only -Werror %s

#ifdef __DEPRECATED
#warning This file is deprecated. // expected-warning {{This file is deprecated.}}
#endif
