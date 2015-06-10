// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-default-args -std=c++11 %s

template<typename T> struct A;
template<typename T> struct B;
template<typename T> struct C;
template<typename T = int> struct D;
template<typename T = int> struct E {};
template<typename T> struct H {}; // expected-note {{here}}

#include "b.h"

template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T = int> struct B;
template<typename T = int> struct C;
template<typename T> struct D {};
template<typename T> struct F {};
template<typename T> struct G {}; // expected-note {{here}}

#include "c.h"

A<> a;
B<> b;
extern C<> c;
D<> d;
E<> e;
F<> f;
G<> g; // expected-error {{too few}}
H<> h; // expected-error {{too few}}
