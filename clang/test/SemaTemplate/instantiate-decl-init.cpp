// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5426 - the non-dependent obj would be fully processed and wrapped in a
// CXXConstructExpr at definition time, which would lead to a failure at
// instantiation time.
struct arg {
  arg();
};

struct oldstylemove {
  oldstylemove(oldstylemove&);
  oldstylemove(const arg&);
};

template <typename T>
void fn(T t, const arg& arg) {
  oldstylemove obj(arg);
}

void test() {
  fn(1, arg());
}
