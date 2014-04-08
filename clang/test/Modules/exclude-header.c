// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodules-cache-path=%t -I %S/Inputs/exclude-header %s -verify

@import x;

a var_a;
b var_b1; // expected-error {{unknown type name 'b'}}

@import y;

b var_b2;
