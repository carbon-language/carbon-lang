// RUN: mkdir -p %t.dir
// RUN: %clang_cc1 -analyze -analyzer-output=html -analyzer-checker=core -o %t.dir
// RUN: ls %t.dir | not grep report
// RUN: rm -fR %t.dir

// This tests that we do not currently emit HTML diagnostics for reports that
// cross file boundaries.

#include "html-diags-multifile.h"

#define CALL_HAS_BUG(q) has_bug(q)

void test_call_macro() {
  CALL_HAS_BUG(0);
}

