// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'int a, b;' > %t/x.h
// RUN: echo 'module x { header "x.h" module y {} }' > %t/map
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s

int use_1 = a; // expected-error {{undeclared}}

import x;

int use_2 = b; // ok

import x [[]];
import x [[foo]]; // expected-warning {{unknown attribute 'foo' ignored}}
import x [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to a module import}}
import x [[blarg::noreturn]]; // expected-warning {{unknown attribute 'noreturn' ignored}}

import x.y;
import x.; // expected-error {{expected a module name after module import}}
import .x; // expected-error {{expected a module name after module import}}

import blarg; // expected-error {{module 'blarg' not found}}
