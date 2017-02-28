// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.API -analyzer-viz-egraph-ubigraph -verify %s
// expected-no-diagnostics

int f(int x) {
  return x < 0 ? 0 : 42;
}

