// RUN: clang-cc -fsyntax-only -verify %s
// PR5488

struct X {
  int x;
};

struct Iter {
  X* operator->();
};

template <typename T>
void Foo() {
  (void)Iter()->x;
}

void Func() {
  Foo<int>();
}

