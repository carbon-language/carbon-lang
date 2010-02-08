// RUN: %clang_cc1 -fsyntax-only -verify %s

struct meta {
  template<typename U>
  struct apply {
    typedef U* type;
  };
};

template<typename T, typename U>
void f(typename T::template apply<U>::type);

void test_f(int *ip) {
  f<meta, int>(ip);
}
