// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 -D__ALLOW_STDC_ATOMICS_IN_CXX__ %s -verify

#include <stdatomic.h>

#ifndef __ALLOW_STDC_ATOMICS_IN_CXX__
// expected-error@stdatomic.h:* {{<stdatomic.h> is incompatible with the C++ standard library}}
#else
// expected-no-diagnostics
#endif
