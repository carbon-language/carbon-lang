// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=%S/Inputs/modular_maps/modulea.map -I %S/Inputs/modular_maps %s -verify

#include "common.h"
#include "a.h"
#include "b.h" // expected-error {{private header}}
const int v = a + c;
const int val = a + b + c; // expected-error {{undeclared identifier}}
