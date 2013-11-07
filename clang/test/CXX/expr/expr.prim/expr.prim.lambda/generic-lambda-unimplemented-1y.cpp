// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify
//expected-no-diagnostics
namespace lambda_capturing {
// FIXME: Once return type deduction is implemented for generic lambdas
// this will need to be updated.
void test() {
  int i = 10;
  {
    auto L = [=](auto a) -> int { 
      return i + a;
    };
    L(3);
  }
  {
    auto L = [i](auto a) -> int { 
      return i + a;
    };
    L(3);
  }  
  {
    auto L = [i=i](auto a) -> int { 
      return i + a;
    };
    L(3);
  }  

  
}

}

