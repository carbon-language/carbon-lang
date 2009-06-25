// RUN: clang-cc -fsyntax-only %s

template<typename T> struct A { };

template<typename T> A<T> f0(T*);

void test_f0(int *ip, float const *cfp) {
  A<int> a0 = f0(ip);
  A<const float> a1 = f0(cfp);
}

