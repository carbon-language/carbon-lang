// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify

namespace lambda_capturing {
// FIXME: Once return type deduction is implemented for generic lambdas
// this will need to be updated.
void test() {
  int i = 10;
  {
    auto L = [=](auto a) -> int { //expected-error{{unimplemented}}
      return i + a;
    };
    L(3);
  }
  {
    auto L = [i](auto a) -> int { //expected-error{{unimplemented}}
      return i + a;
    };
    L(3);
  }  
  {
    auto L = [i = i](auto a) -> int { //expected-error{{unimplemented}}
      return i + a;
    };
    L(3);
  }  

  
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


