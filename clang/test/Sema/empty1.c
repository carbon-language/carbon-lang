// RUN: %clang_cc1 %s -fsyntax-only -verify -Wc++-compat

struct emp_1 { // expected-warning {{empty struct has size 0 in C, size 1 in C++}}
};

union emp_2 { // expected-warning {{empty union has size 0 in C, size 1 in C++}}
};

struct emp_3 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  int : 0;
};

union emp_4 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  int : 0;
};

struct emp_5 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  int : 0;
  int : 0;
};

union emp_6 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  int : 0;
  int : 0;
};

struct emp_7 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  struct emp_1 f1;
};

union emp_8 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  struct emp_1 f1;
};

struct emp_9 { // expected-warning {{struct has size 0 in C, non-zero size in C++}}
  struct emp_1 f1;
  union emp_2 f2;
};
