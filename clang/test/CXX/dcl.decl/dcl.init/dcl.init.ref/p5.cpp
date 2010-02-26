// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5909 {
  struct Foo {
    int x : 20;
  };
  
  bool Test(const int& foo);
  
  const Foo f = { 0 };  // It compiles without the 'const'.
  bool z = Test(f.x);
}

namespace PR6264 {
  typedef int (&T)[3];
  struct S
  {
    operator T ();
  };
  void f()
  {
    T bar = S();
  }
}
