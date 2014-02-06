// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fmodules-cache-path=%t -I %S/Inputs %s

#include "linkage-merge-bar.h"

static int f(int);
int f(int);

static void g(int);
// expected-error@-1 {{functions that differ only in their return type cannot be overloaded}}
// expected-note@Inputs/linkage-merge-foo.h:2 {{previous declaration is here}}
