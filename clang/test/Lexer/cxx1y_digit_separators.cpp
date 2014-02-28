// RUN: %clang_cc1 -std=c++1y -verify %s

int operator""ms(unsigned long long); // expected-warning {{reserved}}
float operator""ms(long double); // expected-warning {{reserved}}

int operator""_foo(unsigned long long);

namespace integral {
  static_assert(1'2'3 == 12'3, "");
  static_assert(1'000'000 == 0xf'4240, "");
  static_assert(0'004'000'000 == 0x10'0000, "");
  static_assert(0b0101'0100 == 0x54, "");

  int a = 123'; //'; // expected-error {{expected ';'}}
  int b = 0'xff; // expected-error {{digit separator cannot appear at end of digit sequence}} expected-error {{suffix 'xff' on integer}}
  int c = 0x'ff; // expected-error {{suffix 'x'ff' on integer}}
  int d = 0'1234; // ok, octal
  int e = 0'b1010; // expected-error {{digit 'b' in octal constant}}
  int f = 0b'1010; // expected-error {{invalid digit 'b' in octal}}
  int g = 123'ms; // expected-error {{digit separator cannot appear at end of digit sequence}}
  int h = 0x1e+1; // expected-error {{invalid suffix '+1' on integer constant}}
  int i = 0x1'e+1; // ok, 'e+' is not recognized after a digit separator

  int z = 0'123'_foo; //'; // expected-error {{cannot appear at end of digit seq}}
}

namespace floating {
  static_assert(0'123.456'7 == 123.4567, "");
  static_assert(1e1'0 == 10'000'000'000, "");

  float a = 1'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float b = 1'0e1;
  float c = 1.'0e1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float d = 1.0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float e = 1e'1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float f = 1e1'ms; // expected-error {{digit separator cannot appear at end of digit sequence}}
}

#line 123'456
static_assert(__LINE__ == 123456, "");

// x has value 0 in C++11 and 34 in C++1y.
#define M(x, ...) __VA_ARGS__
constexpr int x = { M(1'2,3'4) };
static_assert(x == 34, "");
