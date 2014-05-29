// This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
// and -Rpass-analysis) with the inliner. The test is designed to
// always trigger the inliner, so it should be independent of the
// optimization level.

// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O0 -gline-tables-only -emit-obj -verify -S -o /dev/null 2> %t.err

// RUN: %clang -c %s -Rpass=inline -O0 -S -o /dev/null 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=INLINE-NO-LOC

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

float foz(int x, int y) __attribute__((noinline));
float foz(int x, int y) { return x * y; }

// The negative diagnostics are emitted twice because the inliner runs
// twice.
//
// expected-remark@+6 {{foz should never be inlined (cost=never)}}
// expected-remark@+5 {{foz will not be inlined into bar}}
// expected-remark@+4 {{foz should never be inlined}}
// expected-remark@+3 {{foz will not be inlined into bar}}
// expected-remark@+2 {{foo should always be inlined}}
// expected-remark@+1 {{foo inlined into bar}}
int bar(int j) { return foo(j, j - 2) * foz(j - 2, j); }

// INLINE-NO-LOC: {{^remark: foo inlined into bar}}
// INLINE-NO-LOC: note: use -gline-tables-only -gcolumn-info to track
