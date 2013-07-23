// RUN: %clang_cc1 -std=c++1y %s -include %s -verify

#ifndef INCLUDED
#define INCLUDED

#pragma clang system_header
namespace std {
  using size_t = decltype(sizeof(0));

  struct duration {};
  duration operator"" ns(unsigned long long);
  duration operator"" us(unsigned long long);
  duration operator"" ms(unsigned long long);
  duration operator"" s(unsigned long long);
  duration operator"" min(unsigned long long);
  duration operator"" h(unsigned long long);

  struct string {};
  string operator"" s(const char*, size_t);
}

#else

using namespace std;
duration a = 1ns, b = 1us, c = 1ms, d = 1s, e = 1min, f = 1h;
string s = "foo"s;
char error = 'x's; // expected-error {{invalid suffix}} expected-error {{expected ';'}}

int _1z = 1z; // expected-error {{invalid suffix}}
int _1b = 1b; // expected-error {{invalid digit}}

#endif
