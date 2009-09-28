// RUN: clang-cc -fsyntax-only -verify %s

extern "C" void * malloc(int);

template <typename T> struct A {
  void *malloc(int);
};

template <typename T>
inline void *A<T>::malloc(int)
{
  return 0;
}

void f() {
  malloc(10);
}
