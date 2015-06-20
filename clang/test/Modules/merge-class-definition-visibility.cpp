// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/merge-class-definition-visibility/modmap \
// RUN:            -I%S/Inputs/merge-class-definition-visibility \
// RUN:            -fmodules-cache-path=%t %s -verify
// expected-no-diagnostics

#include "c.h"
template<typename T> struct X { T t; };
typedef X<A> XA;

#include "d.h"
// Ensure that this triggers the import of the second definition from d.h,
// which is necessary to make the definition of A visible in the template
// instantiation.
XA xa;
