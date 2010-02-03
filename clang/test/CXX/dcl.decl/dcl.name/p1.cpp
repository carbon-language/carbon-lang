// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace pr6200 {
  struct v {};
  struct s {
    int i;
    operator struct v() { return v(); };
  };

  void f()
  {
    // Neither of these is a declaration.
    (void)new struct s;
    (void)&s::operator struct v;
  }
}
