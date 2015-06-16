// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s

#include "linkage-merge-bar.h"

static int f(int);
int f(int);

static void g(int);
// FIXME: Whether we notice the problem here depends on the order in which we
// happen to find lookup results for 'g'; LookupResult::resolveKind needs to
// be taught to prefer a visible result over a non-visible one.
//
// expected-error@9 {{functions that differ only in their return type cannot be overloaded}}
// expected-note@Inputs/linkage-merge-foo.h:2 {{previous declaration is here}}
