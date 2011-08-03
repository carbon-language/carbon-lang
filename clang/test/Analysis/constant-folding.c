// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode -Wno-null-dereference -verify %s

// Trigger a warning if the analyzer reaches this point in the control flow.
#define WARN ((void)*(char*)0)

// There should be no warnings unless otherwise indicated.

void testComparisons (int a) {
  // Sema can already catch the simple comparison a==a,
  // since that's usually a logic error (and not path-dependent).
  int b = a;
  if (!(b==a)) WARN; // expected-warning{{never executed}}
  if (!(b>=a)) WARN; // expected-warning{{never executed}}
  if (!(b<=a)) WARN; // expected-warning{{never executed}}
  if (b!=a) WARN;    // expected-warning{{never executed}}
  if (b>a) WARN;     // expected-warning{{never executed}}
  if (b<a) WARN;     // expected-warning{{never executed}}
}

void testSelfOperations (int a) {
  if ((a|a) != a) WARN; // expected-warning{{never executed}}
  if ((a&a) != a) WARN; // expected-warning{{never executed}}
  if ((a^a) != 0) WARN; // expected-warning{{never executed}}
  if ((a-a) != 0) WARN; // expected-warning{{never executed}}
}

void testIdempotent (int a) {
  if ((a*1) != a) WARN;    // expected-warning{{never executed}}
  if ((a/1) != a) WARN;    // expected-warning{{never executed}}
  if ((a+0) != a) WARN;    // expected-warning{{never executed}}
  if ((a-0) != a) WARN;    // expected-warning{{never executed}}
  if ((a<<0) != a) WARN;   // expected-warning{{never executed}}
  if ((a>>0) != a) WARN;   // expected-warning{{never executed}}
  if ((a^0) != a) WARN;    // expected-warning{{never executed}}
  if ((a&(~0)) != a) WARN; // expected-warning{{never executed}}
  if ((a|0) != a) WARN;    // expected-warning{{never executed}}
}

void testReductionToConstant (int a) {
  if ((a*0) != 0) WARN; // expected-warning{{never executed}}
  if ((a&0) != 0) WARN; // expected-warning{{never executed}}
  if ((a|(~0)) != (~0)) WARN; // expected-warning{{never executed}}
}

void testSymmetricIntSymOperations (int a) {
  if ((2+a) != (a+2)) WARN; // expected-warning{{never executed}}
  if ((2*a) != (a*2)) WARN; // expected-warning{{never executed}}
  if ((2&a) != (a&2)) WARN; // expected-warning{{never executed}}
  if ((2^a) != (a^2)) WARN; // expected-warning{{never executed}}
  if ((2|a) != (a|2)) WARN; // expected-warning{{never executed}}
}

void testAsymmetricIntSymOperations (int a) {
  if (((~0) >> a) != (~0)) WARN; // expected-warning{{never executed}}
  if ((0 >> a) != 0) WARN; // expected-warning{{never executed}}
  if ((0 << a) != 0) WARN; // expected-warning{{never executed}}

  // Unsigned right shift shifts in zeroes.
  if ((((unsigned)(~0)) >> ((unsigned) a)) != ((unsigned)(~0)))
    WARN; // expected-warning{{}}
}

void testLocations (char *a) {
  char *b = a;
  if (!(b==a)) WARN; // expected-warning{{never executed}}
  if (!(b>=a)) WARN; // expected-warning{{never executed}}
  if (!(b<=a)) WARN; // expected-warning{{never executed}}
  if (b!=a) WARN; // expected-warning{{never executed}}
  if (b>a) WARN; // expected-warning{{never executed}}
  if (b<a) WARN; // expected-warning{{never executed}}
  if (b-a) WARN; // expected-warning{{never executed}}
}
