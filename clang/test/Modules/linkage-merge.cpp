// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fmodules-cache-path=%t -I %S/Inputs %s

#include "linkage-merge-bar.h"

static int f(int);
int f(int);

static void g(int);
// expected-error@-1 {{declaration conflicts with target of using declaration already in scope}}
// expected-note@Inputs/linkage-merge-foo.h:2 {{target of using declaration}}
// expected-note@Inputs/linkage-merge-bar.h:3 {{using declaration}}
