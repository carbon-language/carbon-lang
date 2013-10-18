// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -I %S/Inputs/submodules %s -verify
// FIXME: When we have a syntax for modules in C++, use that.

@import std.vector;

vector<int> vi;

// Note: remove_reference is not visible yet.
remove_reference<int&>::type *int_ptr = 0; // expected-error{{declaration of 'remove_reference' must be imported from module 'std.type_traits' before it is required}}
// expected-note@Inputs/submodules/type_traits.h:2{{previous}}
// expected-note@Inputs/submodules/hash_map.h:1{{previous}}

@import std.typetraits; // expected-error{{no submodule named 'typetraits' in module 'std'; did you mean 'type_traits'?}}

vector<float> vf;
remove_reference<int&>::type *int_ptr2 = 0;

@import std.vector.compare; // expected-error{{no submodule named 'compare' in module 'std.vector'}}

@import std; // import everything in 'std'

// hash_map still isn't available.
hash_map<int, float> ints_to_floats; // expected-error{{declaration of 'hash_map' must be imported from module 'std.hash_map' before it is required}}

@import std.hash_map;

hash_map<int, float> ints_to_floats2;

@import import_self.b;
extern MyTypeA import_self_test_a; // expected-error {{must be imported from module 'import_self.a'}}
// expected-note@import-self-a.h:1 {{here}}
extern MyTypeC import_self_test_c;
// FIXME: This should be valid; import_self.b re-exports import_self.d.
extern MyTypeD import_self_test_d; // expected-error {{must be imported from module 'import_self.d'}}
// expected-note@import-self-d.h:1 {{here}}
