// This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
// and -Rpass-analysis) with the inliner. The test is designed to
// always trigger the inliner, so it should be independent of the
// optimization level.

// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O0 -gline-tables-only -emit-llvm-only -verify
// RUN: %clang_cc1 %s -DNDEBUG -Rpass=inline -emit-llvm-only -verify

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

float foz(int x, int y) __attribute__((noinline));
float foz(int x, int y) { return x * y; }

// The negative diagnostics are emitted twice because the inliner runs
// twice.
//
int bar(int j) {
#ifndef NDEBUG
// expected-remark@+7 {{foz should never be inlined (cost=never)}}
// expected-remark@+6 {{foz will not be inlined into bar}}
// expected-remark@+5 {{foz should never be inlined}}
// expected-remark@+4 {{foz will not be inlined into bar}}
// expected-remark@+3 {{foo should always be inlined}}
// expected-remark@+2 {{foo inlined into bar}}
#endif
  return foo(j, j - 2) * foz(j - 2, j);
}
#ifdef NDEBUG
// expected-remark@-2 {{foo inlined into bar}} expected-note@-2 {{use -gline-tables-only -gcolumn-info to track source location information for this optimization remark}}
#endif
