// RUN: %clang_cc1 -analyze -analyzer-checker=unix.Malloc %s
// Do not crash due to division by zero

int f(unsigned int a) {
  if (a <= 0) return 1 / a;
  return a;
}
