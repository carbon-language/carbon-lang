// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5908
template <typename Iterator>
void Test(Iterator it) {
  *(it += 1);
}
