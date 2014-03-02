// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C -DNAMESPACE
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DNAMESPACE

#ifdef NAMESPACE
namespace M {
#endif

#ifdef EXTERN_C
extern "C" {
#endif

#ifdef EXTERN_CXX
extern "C++" {
#endif

#ifdef CXX_HEADER
#define HEADER "cxx-header.h"
#else
#define HEADER "c-header.h"
#endif

#include HEADER

#if defined(EXTERN_C) && !defined(EXTERN_CXX) && defined(CXX_HEADER)
// expected-error@-3 {{import of C++ module 'cxx_library' appears within extern "C" language linkage specification}}
// expected-note@-17 {{extern "C" language linkage specification begins here}}
#elif defined(NAMESPACE)
// expected-error-re@-6 {{import of module '{{c_library|cxx_library}}' appears within namespace 'M'}}
// expected-note@-24 {{namespace 'M' begins here}}
#endif

#ifdef EXTERN_CXX
}
#endif

#ifdef EXTERN_C
}
#endif

#ifdef NAMESPACE
}
using namespace M;
#endif

namespace N {
  int k = f();

  extern "C" {
    int f;
#if !defined(CXX_HEADER)
    // expected-error@-2 {{redefinition of 'f' as different kind of symbol}}
    // expected-note@c-header.h:1 {{previous}}
#endif
  }
}

suppress_expected_no_diagnostics_error; // expected-error {{}}
