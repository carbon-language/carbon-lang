// RUN: %clang_cc1 -triple armv7 -std=c++14 -x c++ %s -fsyntax-only
// expected-no-diagnostics

void deduce() {
  auto lambda = [](int i) __attribute__ (( pcs("aapcs") )) {
    return i;
  };
  lambda(42);
}

