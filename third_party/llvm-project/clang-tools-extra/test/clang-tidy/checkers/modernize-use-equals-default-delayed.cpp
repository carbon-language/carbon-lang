// RUN: clang-tidy %s -checks=-*,modernize-use-equals-default -- -std=c++11 -fdelayed-template-parsing -fexceptions | count 0
// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.
// FIXME: Make the test work in all language modes.

template <typename Ty>
struct S {
  S<Ty>& operator=(const S<Ty>&) { return *this; }
};
