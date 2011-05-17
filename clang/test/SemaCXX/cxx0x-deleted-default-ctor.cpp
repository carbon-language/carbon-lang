// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct non_trivial {
  non_trivial();
  non_trivial(const non_trivial&);
  non_trivial& operator = (const non_trivial&);
  ~non_trivial();
};

union bad_union { // expected-note {{marked deleted here}}
  non_trivial nt;
};
bad_union u; // expected-error {{call to deleted constructor}}
union bad_union2 { // expected-note {{marked deleted here}}
  const int i;
};
bad_union2 u2; // expected-error {{call to deleted constructor}}

struct bad_anon { // expected-note {{marked deleted here}}
  union {
    non_trivial nt;
  };
};
bad_anon a; // expected-error {{call to deleted constructor}}
struct bad_anon2 { // expected-note {{marked deleted here}}
  union {
    const int i;
  };
};
bad_anon2 a2; // expected-error {{call to deleted constructor}}

// This would be great except that we implement
union good_union {
  const int i;
  float f;
};
good_union gu;
struct good_anon {
  union {
    const int i;
    float f;
  };
};
good_anon ga;

struct good : non_trivial {
  non_trivial nt;
};
good g;
