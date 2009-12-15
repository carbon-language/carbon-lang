// RUN: %clang_cc1 -fsyntax-only %s

template<typename T, T I, int J>
struct adder {
  enum {
    value = I + J,
    value2
  };
};

int array1[adder<long, 3, 4>::value == 7? 1 : -1];
