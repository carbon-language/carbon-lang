// RUN: clang-cc -fsyntax-only -verify %s

namespace A {
  void g();
}

namespace X {
  using A::g; 
}

void h()
{
  A::g();
  X::g();
}
