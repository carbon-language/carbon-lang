// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/merge-template-members -verify %s
// expected-no-diagnostics

template<typename> struct A { int n; };
template<typename> struct B { typedef A<void> C; };
template class B<int>;

#include "update.h"
B<int>::C use2;
