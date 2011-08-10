// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s

namespace PR10622 {
  struct foo {
    const int first;
    foo(const foo&) = default;
  };
  void find_or_insert(const foo& __obj) {
    foo x(__obj);
  }

  struct bar : foo {
    bar(const bar&) = default;
  };
  void test_bar(const bar &obj) {
    bar obj2(obj);
  }
}
