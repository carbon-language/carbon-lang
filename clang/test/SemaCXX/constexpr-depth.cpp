// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -DMAX=128 -fconstexpr-depth 128
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -DMAX=1 -fconstexpr-depth 1
// RUN: %clang -std=c++11 -fsyntax-only -Xclang -verify %s -DMAX=10 -fconstexpr-depth=10

constexpr int depth(int n) { return n > 1 ? depth(n-1) : 0; }

constexpr int kBad = depth(MAX + 1); // expected-error {{must be initialized by a constant expression}}
constexpr int kGood = depth(MAX);
