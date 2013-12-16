// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-maps -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/modular_maps/modulea.map -I %S/Inputs/modular_maps %s -verify

// expected-warning@Inputs/modular_maps/modulea.map:4{{header 'doesnotexists.h' not found}}

#include "common.h"
#include "a.h"
#include "b.h" // expected-error {{private header}}
const int v = a + c;
const int val = a + b + c; // expected-error {{undeclared identifier}}
