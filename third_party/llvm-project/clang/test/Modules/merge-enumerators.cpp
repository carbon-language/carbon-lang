// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'namespace N { enum E { A }; }' > %t/a.h
// RUN: echo '#include "a.h"' > %t/b.h
// RUN: touch %t/x.h
// RUN: echo 'module B { module b { header "b.h" } module x { header "x.h" } }' > %t/b.modulemap
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x c++ -fmodule-map-file=%t/b.modulemap %s -I%t -verify -fmodules-local-submodule-visibility
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x c++ -fmodule-map-file=%t/b.modulemap %s -I%t -verify -fmodules-local-submodule-visibility -DMERGE_LATE

#ifndef MERGE_LATE
// expected-no-diagnostics
#include "a.h"
#endif

#include "x.h"

#ifdef MERGE_LATE
namespace N {
  enum { A } a; // expected-note {{candidate}}
  // expected-note@a.h:1 {{candidate}} (from module B.b)
}
#include "a.h"
#endif

N::E e = N::A;
#ifdef MERGE_LATE
// expected-error@-2 {{ambiguous}}
#endif
