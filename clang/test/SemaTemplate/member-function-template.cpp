// RUN: clang-cc -fsyntax-only %s

struct X {
  template<typename T> T& f(T);
};
