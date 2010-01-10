// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s

void f1() {
  int const &i = 3;
  int b = i;

  int *p = 0;

  if (b != 3)
    *p = 1; // no-warning
}
