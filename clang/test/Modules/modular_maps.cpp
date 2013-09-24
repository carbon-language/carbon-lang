// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=%S/Inputs/modular_maps/modulea.map -I %S/Inputs/modular_maps %s -verify

#include "a.h"
#include "b.h" // expected-error {{private header}}
const int val = a + b; // expected-error {{undeclared identifier}}
