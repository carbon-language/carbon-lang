// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Winvalid-noreturn %s -verify

template<typename T>
void test_attributes() {
  auto nrl = []() [[noreturn]] {}; // expected-warning{{function declared 'noreturn' should not return}}
}

template void test_attributes<int>(); // expected-note{{in instantiation of function}}

template<typename T>
void call_with_zero() {
  [](T *ptr) -> T& { return *ptr; }(0);
}

template void call_with_zero<int>();

template<typename T>
T captures(T x, T y) {
  auto lambda = [=, &y] () -> T {
    T i = x;
    return i + y;
  };

  return lambda();
}

struct X {
  X(const X&);
};

X operator+(X, X);
X operator-(X, X);

template int captures(int, int);
template X captures(X, X);

template<typename T>
int infer_result(T x, T y) {
  auto lambda = [=](bool b) { return x + y; };
  return lambda(true); // expected-error{{no viable conversion from 'X' to 'int'}}
}

template int infer_result(int, int);
template int infer_result(X, X); // expected-note{{in instantiation of function template specialization 'infer_result<X>' requested here}}

