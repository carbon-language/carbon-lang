// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR8345
template<typename T> T f(T* value) {
  return __sync_add_and_fetch(value, 1);
}
int g(long long* x) { return f(x); }
int g(int* x) { return f(x); }
