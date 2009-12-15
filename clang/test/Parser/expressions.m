// RUN: %clang_cc1 -parse-noop %s

void test1() {
  @"s";            // expected-warning {{expression result unused}}
}

