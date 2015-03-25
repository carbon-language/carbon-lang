#ifndef STRESS1_COMMON_H
#define STRESS1_COMMON_H

inline char function00(char x) { return x + x; }
inline short function00(short x) { return x + x; }
inline int function00(int x) { return x + x; }

namespace N00 {
struct S00 {
  char c;
  short s;
  int i;

  S00(char x) : c(x) {}
  S00(short x) : s(x) {}
  S00(int x) : i(x) {}

  char method00(char x) { return x + x; }
  short method00(short x) { return x + x; }
  int method00(int x) { return x + x; }

  operator char() { return c; }
  operator short() { return s; }
  operator int() { return i; }
};
struct S01 {};
struct S03 {};
}

namespace N01 {
struct S00 : N00::S00 {
  using N00::S00::S00;
};
struct S01 {};
struct S02 {};
}

#endif
