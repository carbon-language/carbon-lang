// RUN: %clang_cc1 -analyze -analyzer-checker=experimental.core.SizeofPtr -verify %s

struct s {
};

int f(struct s *p) {
  return sizeof(p); // expected-warning{{The code calls sizeof() on a pointer type. This can produce an unexpected result.}}
}
