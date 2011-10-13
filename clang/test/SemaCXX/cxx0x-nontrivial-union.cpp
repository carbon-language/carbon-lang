// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct non_trivial {
  non_trivial();
  non_trivial(const non_trivial&);
  non_trivial& operator = (const non_trivial&);
  ~non_trivial();
};

union u {
  non_trivial nt;
};

union bad {
  static int i; // expected-error {{static data member}}
};

struct s {
  union {
    non_trivial nt;
  };
};
