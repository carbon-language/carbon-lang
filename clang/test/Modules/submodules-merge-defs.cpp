// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules-cache-path=%t -fmodules -I %S/Inputs/submodules-merge-defs %s -verify -fno-modules-error-recovery -DTEXTUAL
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules-cache-path=%t -fmodules -I %S/Inputs/submodules-merge-defs %s -verify -fno-modules-error-recovery

// Trigger import of definitions, but don't make them visible.
#include "empty.h"

A pre_a; // expected-error {{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:1 +{{here}}
// FIXME: We should warn that use_a is being used without being imported.
int pre_use_a = use_a(pre_a); // expected-error {{'A' must be imported}}

B::Inner2 pre_bi; // expected-error +{{must be imported}}
// expected-note@defs.h:4 +{{here}}
// expected-note@defs.h:11 +{{here}}

C_Base<1> pre_cb1; // expected-error +{{must be imported}}
// expected-note@defs.h:15 +{{here}}
C1 pre_c1; // expected-error +{{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:17 +{{here}}
C2 pre_c2; // expected-error +{{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:18 +{{here}}

D::X pre_dx; // expected-error +{{must be imported}}
// expected-note@defs.h:20 +{{here}}
// expected-note@defs.h:21 +{{here}}
// FIXME: We should warn that use_dx is being used without being imported.
int pre_use_dx = use_dx(pre_dx);

int pre_e = E(0); // expected-error {{must be imported}}
// expected-note@defs.h:24 +{{here}}

int pre_ff = F<int>().f(); // expected-error +{{must be imported}}
int pre_fg = F<int>().g<int>(); // expected-error +{{must be imported}}
// expected-note@defs.h:26 +{{here}}

// Make definitions from second module visible.
#ifdef TEXTUAL
#include "import-and-redefine.h"
#else
#include "merged-defs.h"
#endif

A post_a;
int post_use_a = use_a(post_a);
B::Inner2 post_bi;
C_Base<1> post_cb1;
C1 c1;
C2 c2;
D::X post_dx;
int post_use_dx = use_dx(post_dx);
int post_e = E(0);
int post_ff = F<char>().f();
int post_fg = F<char>().g<int>();
