// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s

void f1() {
  int const &i = 3;
  int b = i;
}
