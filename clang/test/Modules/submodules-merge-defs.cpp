// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules-cache-path=%t -fmodules -I %S/Inputs/submodules-merge-defs %s -verify -fno-modules-error-recovery

// Trigger import of definitions, but don't make them visible.
#include "empty.h"

A pre_a; // expected-error {{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:1 {{here}}

B::Inner2 pre_bi; // expected-error +{{must be imported}}
// expected-note@defs.h:4 +{{here}}
// expected-note@defs.h:10 +{{here}}

C_Base<1> pre_cb1; // expected-error +{{must be imported}}
// expected-note@defs.h:13 +{{here}}
C1 pre_c1; // expected-error +{{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:15 +{{here}}
C2 pre_c2; // expected-error +{{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:16 +{{here}}

D::X pre_dx; // expected-error +{{must be imported}}
// expected-note@defs.h:18 +{{here}}
// expected-note@defs.h:19 +{{here}}

// Make definitions from second module visible.
#include "import-and-redefine.h"

A post_a;
B::Inner2 post_bi;
C_Base<1> post_cb1;
C1 c1;
C2 c2;
D::X post_dx;
