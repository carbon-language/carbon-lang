// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-experimental-checks -verify %s

// Trigger a warning if the analyzer reaches this point in the control flow.
#define WARN ((void)*(char*)0)

// There should be no warnings unless otherwise indicated.

void testComparisons (int a) {
  // Sema can already catch the simple comparison a==a,
  // since that's usually a logic error (and not path-dependent).
  int b = a;
  if (!(b==a)) WARN;
  if (!(b>=a)) WARN;
  if (!(b<=a)) WARN;
  if (b!=a) WARN;
  if (b>a) WARN;
  if (b<a) WARN;
}

void testSelfOperations (int a) {
  if ((a|a) != a) WARN;
  if ((a&a) != a) WARN;
  if ((a^a) != 0) WARN;
  if ((a-a) != 0) WARN;
}

void testIdempotent (int a) {
  if ((a*1) != a) WARN;
  if ((a/1) != a) WARN;
  if ((a+0) != a) WARN;
  if ((a-0) != a) WARN;
  if ((a<<0) != a) WARN;
  if ((a>>0) != a) WARN;
  if ((a^0) != a) WARN;
  if ((a&(~0)) != a) WARN;
  if ((a|0) != a) WARN;
}

void testReductionToConstant (int a) {
  if ((a*0) != 0) WARN;
  if ((a&0) != 0) WARN;
  if ((a|(~0)) != (~0)) WARN;
}

void testSymmetricIntSymOperations (int a) {
  if ((2+a) != (a+2)) WARN;
  if ((2*a) != (a*2)) WARN;
  if ((2&a) != (a&2)) WARN;
  if ((2^a) != (a^2)) WARN;
  if ((2|a) != (a|2)) WARN;
}

void testAsymmetricIntSymOperations (int a) {
  if (((~0) >> a) != (~0)) WARN;
  if ((0 >> a) != 0) WARN;
  if ((0 << a) != 0) WARN;

  // Unsigned right shift shifts in zeroes.
  if ((((unsigned)(~0)) >> ((unsigned) a)) != ((unsigned)(~0)))
    WARN; // expected-warning{{}}
}
