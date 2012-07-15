// RUN: %clang_cc1 %s -fsyntax-only -verify -Wunique-enum
enum A { A1 = 1, A2 = 1, A3 = 1 };  // expected-warning {{all elements of 'A' are initialized with literals to value 1}} \
// expected-note {{initialize the last element with the previous element to silence this warning}}
enum { B1 = 1, B2 = 1, B3 = 1 }; // no warning
enum C { // expected-warning {{all elements of 'C' are initialized with literals to value 1}}
  C1 = true,
  C2 = true  // expected-note {{initialize the last element with the previous element to silence this warning}}
};
enum D { D1 = 5, D2 = 5L, D3 = 5UL, D4 = 5LL, D5 = 5ULL };  // expected-warning {{all elements of 'D' are initialized with literals to value 5}} \
// expected-note {{initialize the last element with the previous element to silence this warning}}

// Don't warn on enums with less than 2 elements.
enum E { E1 = 4 };
enum F { F1 };
enum G {};

// Don't warn when integer literals do not initialize the elements.
enum H { H1 = 4, H_MAX = H1, H_MIN = H1 };
enum I { I1 = H1, I2 = 4 };
enum J { J1 = 4, J2 = I2 };
enum K { K1, K2, K3, K4 };

// Don't crash or warn on this one.
// rdar://11875995
enum L {
  L1 = 0x8000000000000000ULL, L2 = 0x0000000000000001ULL
};
