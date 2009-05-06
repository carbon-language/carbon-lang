// RUN: clang-cc -fsyntax-only %s

class C {
  friend class D;
};
