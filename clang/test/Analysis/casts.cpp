// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -verify %s
// expected-no-diagnostics

bool PR14634(int x) {
  double y = (double)x;
  return !y;
}

bool PR14634_implicit(int x) {
  double y = (double)x;
  return y;
}
