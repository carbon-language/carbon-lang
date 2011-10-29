// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace StdExample {

constexpr int f(void *) { return 0; }
constexpr int f(...) { return 1; }
constexpr int g1() { return f(0); }
constexpr int g2(int n) { return f(n); }
constexpr int g3(int n) { return f(n*0); }

namespace N {
  constexpr int c = 5;
  constexpr int h() { return c; }
}
constexpr int c = 0;
constexpr int g4() { return N::h(); }

// FIXME: constexpr calls aren't recognized as ICEs yet, just as foldable.
#define JOIN2(a, b) a ## b
#define JOIN(a, b) JOIN2(a, b)
#define CHECK(n, m) using JOIN(A, __LINE__) = int[n]; using JOIN(A, __LINE__) = int[m];
CHECK(f(0), 0)
CHECK(f('0'), 1)
CHECK(g1(), 0)
CHECK(g2(0), 1)
CHECK(g2(1), 1)
CHECK(g3(0), 1)
CHECK(g3(1), 1)
CHECK(N::h(), 5)
CHECK(g4(), 5)

}
