// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules-cache-path=%t -fmodules -I %S/Inputs/submodules-merge-defs %s -verify -fno-modules-error-recovery

// Trigger import of definitions, but don't make them visible.
#include "empty.h"

A pre_a; // expected-error {{must be imported}} expected-error {{must use 'struct'}}
// expected-note@defs.h:1 {{here}}

// Make definitions from second module visible.
#include "import-and-redefine.h"

A post_a;
