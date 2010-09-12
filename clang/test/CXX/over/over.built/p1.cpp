// RUN: %clang_cc1 -fsyntax-only -verify %s

enum E1 { one };
enum E2 { two };

bool operator >= (E1, E1) {
  return false;
}

bool operator >= (E1, const E2) {
  return false;
}

bool test(E1 a, E1 b, E2 c) {
  return a >= b || a >= c;
}
