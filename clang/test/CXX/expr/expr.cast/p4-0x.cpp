// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct X { };
struct Y : X { };

void test_lvalue_to_rvalue_drop_cvquals(const X &x, const Y &y, const int &i) {
  (void)(X&&)x;
  (void)(int&&)i;
  (void)(X&&)y;
  (void)(Y&&)x;
}
