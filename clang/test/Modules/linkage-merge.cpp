// FIXME: we should be able to put these in the .h file :-(
// expected-note {{target of using declaration}}
// expected-note {{using declaration}}

#include "linkage-merge-bar.h"

static int f(int);
int f(int);

static void g(int); // expected-error {{declaration conflicts with target of using declaration already in scope}}

// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fmodules-cache-path=%t -I %S/Inputs %s
