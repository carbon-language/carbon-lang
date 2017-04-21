// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'export module x; int a, b;' > %t/x.cppm
// RUN: echo 'export module x.y; int c;' > %t/x.y.cppm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface -fmodule-file=%t/x.pcm %t/x.y.cppm -o %t/x.y.pcm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=1 -DEXPORT= -DPARTITION= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=2 -DEXPORT= -DPARTITION= -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=3 -DEXPORT=export -DPARTITION= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=4 -DEXPORT=export -DPARTITION=partition -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=5 -DEXPORT= -DPARTITION=elderberry -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=1 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=6 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[fancy]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=7 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[maybe_unused]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=8 -DEXPORT= -DPARTITION=partition -DMODULE_NAME=z

EXPORT module PARTITION MODULE_NAME;
#if TEST == 3 || TEST == 4
// ok, building object code for module interface
#elif TEST == 5
// expected-error@-4 {{expected ';'}} expected-error@-4 {{requires a type specifier}}
#elif TEST == 6
// expected-warning@-6 {{unknown attribute 'fancy' ignored}}
#elif TEST == 7
// expected-error-re@-8 {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
#elif TEST == 8
// expected-error@-10 {{module partition must be declared 'export'}}
#endif

int use_1 = a;
#if TEST != 2
// expected-error@-2 {{declaration of 'a' must be imported from module 'x' before it is required}}
// expected-note@x.cppm:1 {{here}}
#endif

import x;

int use_2 = b; // ok

import x [[]];
import x [[foo]]; // expected-warning {{unknown attribute 'foo' ignored}}
import x [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to a module import}}
import x [[blarg::noreturn]]; // expected-warning {{unknown attribute 'noreturn' ignored}}

import x.y;
import x.; // expected-error {{expected a module name after 'import'}}
import .x; // expected-error {{expected a module name after 'import'}}

import blarg; // expected-error {{module 'blarg' not found}}
