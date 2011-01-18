// RUN: %clang_cc1 %s -Wunused-macros -Dfoo -Dfoo -verify

#define unused // expected-warning {{macro is not used}}
#define unused
unused
