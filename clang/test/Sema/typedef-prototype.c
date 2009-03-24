// RUN: clang-cc -fsyntax-only -verify %s

typedef int unary_int_func(int arg);
unary_int_func add_one;

int add_one(int arg) {
  return arg + 1;
}
