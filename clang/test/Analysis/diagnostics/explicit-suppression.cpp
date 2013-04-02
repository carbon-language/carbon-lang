// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=true -DSUPPRESSED=1 -verify %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

#include "../Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

void testCopyNull(int *I, int *E) {
  std::copy(I, E, (int *)0);
#ifndef SUPPRESSED
  // This line number comes from system-header-simulator-cxx.h.
  // expected-warning@79 {{Dereference of null pointer}}
#endif
}



























































// PR15613: expected-* can't refer to diagnostics in other source files.
// The current implementation only matches line numbers, but has an upper limit
// of the number of lines in the main source file.
