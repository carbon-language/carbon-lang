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

union static_data_member {
  static int i;
};
int static_data_member::i;

union bad {
  int &i; // expected-error {{union member 'i' has reference type 'int &'}}
};

struct s {
  union {
    non_trivial nt;
  };
};
