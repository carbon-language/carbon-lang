// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

// prvalue
void prvalue() {
  auto&& x = []()->void { }; // expected-error{{lambda expressions are not supported yet}}
  auto& y = []()->void { }; // expected-error{{cannot bind to a temporary of type}} \
  // expected-error{{lambda expressions are not supported yet}}
}

namespace std {
  class type_info;
}

struct P {
  virtual ~P();
};

void unevaluated_operand(P &p, int i) {
  int i2 = sizeof([]()->void{}()); // expected-error{{lambda expression in an unevaluated operand}} \
  // expected-error{{lambda expressions are not supported yet}}
  const std::type_info &ti1 = typeid([&]() -> P& { return p; }()); // expected-error{{lambda expressions are not supported yet}}
  const std::type_info &ti2 = typeid([&]() -> int { return i; }());  // expected-error{{lambda expression in an unevaluated operand}} \
  // expected-error{{lambda expressions are not supported yet}}
}

template<typename T>
struct Boom {
  Boom(const Boom&) { 
    T* x = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}} \
    // expected-error{{cannot initialize a variable of type 'float *' with an rvalue of type 'int'}} \
    // expected-error{{cannot initialize a variable of type 'double *' with an rvalue of type 'int'}}
  }
  void tickle() const;
};

void odr_used(P &p, Boom<int> boom_int, Boom<float> boom_float,
              Boom<double> boom_double) {
  const std::type_info &ti1
    = typeid([=,&p]() -> P& { boom_int.tickle(); return p; }()); // expected-error{{lambda expressions are not supported yet}} \
  // expected-note{{in instantiation of member function 'Boom<int>::Boom' requested here}}
  const std::type_info &ti2
    = typeid([=]() -> int { boom_float.tickle(); return 0; }()); // expected-error{{lambda expression in an unevaluated operand}} \
  // expected-error{{lambda expressions are not supported yet}} \
  // expected-note{{in instantiation of member function 'Boom<float>::Boom' requested here}}

  auto foo = [=]() -> int { boom_double.tickle(); return 0; }; // expected-error{{lambda expressions are not supported yet}} \
  // expected-note{{in instantiation of member function 'Boom<double>::Boom' requested here}}
}
