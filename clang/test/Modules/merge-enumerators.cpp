// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'namespace N { enum E { A }; }' > %t/a.h
// RUN: echo '#include "a.h"' > %t/b.h
// RUN: touch %t/x.h
// RUN: echo 'module B { module b { header "b.h" } module x { header "x.h" } }' > %t/b.modulemap
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x c++ -fmodule-map-file=%t/b.modulemap %s -I%t -verify
// expected-no-diagnostics
#include "a.h"
#include "x.h"
N::E e = N::A;
