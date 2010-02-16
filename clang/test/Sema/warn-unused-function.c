// RUN: %clang_cc1 -fsyntax-only -Wunused-function -verify %s
// XFAIL: *

void foo() {}
static void f2() {} 
static void f1() {f2();} // expected-warning{{unused}}

static int f0() { return 17; } // expected-warning{{unused}}
int x = sizeof(f0());

static void f3();
extern void f3() { } // expected-warning{{unused}}

// FIXME: This will trigger a warning when it should not.
// Update once PR6281 is fixed.
//inline static void f4();
//void f4() { } 
