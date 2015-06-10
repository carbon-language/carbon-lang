// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-default-args -std=c++11 %s
//
// expected-no-diagnostics

template<typename T> struct A;
template<typename T> struct B;
template<typename T> struct C;
template<typename T = int> struct D;

#include "b.h"

template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T = int> struct B;
template<typename T = int> struct C;
template<typename T> struct D {};

A<> a;
B<> b;
extern C<> c;
D<> d;
