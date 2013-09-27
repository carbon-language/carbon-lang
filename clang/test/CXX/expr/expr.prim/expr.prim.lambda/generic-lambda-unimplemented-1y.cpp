// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify
namespace return_type_deduction_ok {
 auto l = [](auto a) ->auto { return a; }(2); 
 auto l2 = [](auto a) ->decltype(auto) { return a; }(2);  
 auto l3 = [](auto a) { return a; }(2); 

}

namespace lambda_capturing {
// FIXME: Once return type deduction is implemented for generic lambdas
// this will need to be updated.
void test() {
  int i = 10;
  auto L = [=](auto a) -> int { //expected-error{{unimplemented}}
    return i + a;
  };
  L(3); 
}

}

namespace nested_generic_lambdas {
void test() {
  auto L = [](auto a) -> int {
    auto M = [](auto b, decltype(a) b2) -> int { //expected-error{{unimplemented}}
      return 1;
    };
    M(a, a);
  };
  L(3); //expected-note{{in instantiation of}}
}
template<class T> void foo(T) {
 auto L = [](auto a) { return a; }; //expected-error{{unimplemented}}
}
template void foo(int); //expected-note{{in instantiation of}}
}

namespace conversion_operator {
void test() {
    auto L = [](auto a) -> int { return a; };
    int (*fp)(int) = L;  //expected-error{{no viable conversion}}
  }
}

namespace generic_lambda_as_default_argument_ok {
  void test(int i = [](auto a)->int { return a; }(3)) {
  
  }
  
}

