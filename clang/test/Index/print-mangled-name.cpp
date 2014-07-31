// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-pch %s -o %t_linux.ast
// RUN: c-index-test -test-print-mangle %t_linux.ast | FileCheck %s --check-prefix=ITANIUM

// RUN: %clang_cc1 -triple i686-pc-win32 -emit-pch %s -o %t_msft.ast
// RUN: c-index-test -test-print-mangle %t_msft.ast | FileCheck %s --check-prefix=MICROSOFT

int foo(int, int);
// ITANIUM: mangled=_Z3fooii
// MICROSOFT: mangled=?foo@@YAHHH

int foo(float, int);
// ITANIUM: mangled=_Z3foofi
// MICROSOFT: mangled=?foo@@YAHMH

struct S {
  int x, y;
};
// ITANIUM: StructDecl{{.*}}mangled=]
// MICROSOFT: StructDecl{{.*}}mangled=]

int foo(S, S&);
// ITANIUM: mangled=_Z3foo1SRS
// MICROSOFT: mangled=?foo@@YAHUS
