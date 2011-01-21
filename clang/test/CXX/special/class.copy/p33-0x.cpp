// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s
class X {
  X(const X&);

public:
  X();
  X(X&&);
};

X return_by_move(int i, X x) {
  X x2;
  if (i == 0)
    return x;
  else if (i == 1)
    return x2;
  else
    return x;
}

void throw_move_only(X x) {
  X x2;
  throw x;
  throw x2;
}
  
