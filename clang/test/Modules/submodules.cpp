// RUN: rm -rf %t
// RUN: %clang_cc1 -Eonly -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/submodules %s -verify
// RUN: %clang_cc1 -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/submodules %s -verify

__import_module__ std.vector;
__import_module__ std.typetraits; // expected-error{{no submodule named 'typetraits' in module 'std'; did you mean 'type_traits'?}}
__import_module__ std.vector.compare; // expected-error{{no submodule named 'compare' in module 'std.vector'}}
