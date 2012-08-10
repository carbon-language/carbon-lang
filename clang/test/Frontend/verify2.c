#if 0
// RUN: %clang_cc1 -verify %s 2>&1 | FileCheck %s

// Please note that all comments are inside "#if 0" blocks so that
// VerifyDiagnosticConsumer sees no comments while processing this
// test-case.
#endif

#include "verify2.h"
#error source

#if 0
// expected-error {{should be ignored}}

//      CHECK: error: 'error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 1: header
// CHECK-NEXT:   Line 10: source
// CHECK-NEXT: 2 errors generated.
#endif
