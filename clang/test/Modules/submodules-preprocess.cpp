// RUN: rm -rf %t
// RUN: %clang_cc1 -Eonly -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/submodules %s -verify

__import_module__ std.vector;

vector<int> vi;
remove_reference<int&>::type *int_ptr = 0;

__import_module__ std.typetraits; // expected-error{{no submodule named 'typetraits' in module 'std'; did you mean 'type_traits'?}}

vector<float> vf;
remove_reference<int&>::type *int_ptr2 = 0;

__import_module__ std.vector.compare; // expected-error{{no submodule named 'compare' in module 'std.vector'}}
