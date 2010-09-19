// RUN: %clang_cc1 -Wunused-value -fsyntax-only -verify %s

void test1() {
  @"s";            // expected-warning {{expression result unused}}
}

