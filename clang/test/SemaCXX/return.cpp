// RUN: clang-cc %s -fsyntax-only -verify

int test1() {
  throw;
}

// PR5071
template<typename T> T f() { }

template<typename T>
void g(T t) {
  return t * 2; // okay
}

template<typename T>
T h() {
  return 17;
}
