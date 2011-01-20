// RUN: %clang_cc1 -fsyntax-only  %s
struct bar {
  enum xxx {
    yyy = sizeof(struct foo*)
  };
  foo *xxx();
};
