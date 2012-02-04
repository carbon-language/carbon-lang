// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -triple=i686-pc-linux-gnu

void ugly_news(int *ip) {
  // These are ill-formed according to one reading of C++98, and at the least
  // have undefined behavior. But they're well-formed, and defined to throw
  // std::bad_array_new_length, in C++11.
  (void)new int[-1]; // expected-warning {{array size is negative}}
  (void)new int[2000000000]; // expected-warning {{array is too large}}
}
