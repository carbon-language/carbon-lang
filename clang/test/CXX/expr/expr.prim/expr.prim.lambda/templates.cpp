// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Winvalid-noreturn %s -verify

template<typename T>
void test_attributes() {
  auto nrl = []() [[noreturn]] {}; // expected-error{{lambda declared 'noreturn' should not return}}
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

// Make sure that lambda's operator() can be used from templates.
template<typename F>
void accept_lambda(F f) {
  f(1);
}

template<typename T>
void pass_lambda(T x) {
  accept_lambda([&x](T y) { return x + y; });
}

template void pass_lambda(int);

namespace std {
  class type_info;
}

namespace p2 {
  struct P {
    virtual ~P();
  };

  template<typename T>
  struct Boom {
    Boom(const Boom&) { 
      T* x = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}} \
      // expected-error{{cannot initialize a variable of type 'float *' with an rvalue of type 'int'}}
    }
    void tickle() const;
  };
  
  template<typename R, typename T>
  void odr_used(R &r, Boom<T> boom) {
    const std::type_info &ti
      = typeid([=,&r] () -> R& { // expected-error{{lambda expression in an unevaluated operand}}
          boom.tickle(); // expected-note{{in instantiation of member function}}
          return r; 
        }()); 
  }

  template void odr_used(int&, Boom<int>); // expected-note{{in instantiation of function template specialization}}

  template<typename R, typename T>
  void odr_used2(R &r, Boom<T> boom) {
    const std::type_info &ti
      = typeid([=,&r] () -> R& {
          boom.tickle(); // expected-note{{in instantiation of member function}}
          return r; 
        }()); 
  }

  template void odr_used2(P&, Boom<float>);
}

namespace p5 {
  struct NonConstCopy {
    NonConstCopy(const NonConstCopy&) = delete;
    NonConstCopy(NonConstCopy&);
  };

  template<typename T>
  void double_capture(T &nc) {
    [=] () mutable {
      [=] () mutable {
        T nc2(nc);
      }();
    }();
  }

  template void double_capture(NonConstCopy&);
}
