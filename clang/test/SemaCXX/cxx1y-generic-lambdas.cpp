// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks %s
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s -DDELAYED_TEMPLATE_PARSING
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fms-extensions %s -DMS_EXTENSIONS
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing -fms-extensions %s -DMS_EXTENSIONS -DDELAYED_TEMPLATE_PARSING

namespace explicit_call {
int test() {
  auto L = [](auto a) { return a; };
  L.operator()(3);
  L.operator()<char>(3.14); //expected-warning{{implicit conversion}}
  return 0;
}  
} //end ns

namespace test_conversion_to_fptr {

void f1(int (*)(int)) { }
void f2(char (*)(int)) { } // expected-note{{candidate}}
void g(int (*)(int)) { } // #1 expected-note{{candidate}}
void g(char (*)(char)) { } // #2 expected-note{{candidate}}
void h(int (*)(int)) { } // #3
void h(char (*)(int)) { } // #4

int test() {
{
  auto glambda = [](auto a) { return a; };
  glambda(1);
  f1(glambda); // OK
  f2(glambda); // expected-error{{no matching function}}
  g(glambda); // expected-error{{call to 'g' is ambiguous}}
  h(glambda); // OK: calls #3 since it is convertible from ID
  
  int& (*fpi)(int*) = [](auto* a) -> auto& { return *a; }; // OK
  
}
{
  
  auto L = [](auto a) { return a; };
  int (*fp)(int) = L;
  fp(5);
  L(3);
  char (*fc)(char) = L;
  fc('b');
  L('c');
  double (*fd)(double) = L;
  fd(3.14);
  fd(6.26);
  L(4.25);
}
{
  auto L = [](auto a) ->int { return a; }; //expected-note 2{{candidate template ignored}}
  int (*fp)(int) = L;
  char (*fc)(char) = L; //expected-error{{no viable conversion}}
  double (*fd)(double) = L; //expected-error{{no viable conversion}}
}

}

namespace more_converion_to_ptr_to_function_tests {


int test() {
  {
    int& (*fpi)(int*) = [](auto* a) -> auto& { return *a; }; // OK
    int (*fp2)(int) = [](auto b) -> int {  return b; };
    int (*fp3)(char) = [](auto c) -> int { return c; };
    char (*fp4)(int) = [](auto d) { return d; }; //expected-error{{no viable conversion}}\
                                                 //expected-note{{candidate template ignored}}
    char (*fp5)(char) = [](auto e) -> int { return e; }; //expected-error{{no viable conversion}}\
                                                 //expected-note{{candidate template ignored}}

    fp2(3);
    fp3('\n');
    fp3('a');
    return 0;
  }
} // end test()

template<class ... Ts> void vfun(Ts ... ) { }

int variadic_test() {

 int (*fp)(int, char, double) = [](auto ... a) -> int { vfun(a...); return 4; };
 fp(3, '4', 3.14);
 
 int (*fp2)(int, char, double) = [](auto ... a) { vfun(a...); return 4; };
 fp(3, '4', 3.14);
 return 2;
}

} // end ns

namespace conversion_operator {
void test() {
    auto L = [](auto a) -> int { return a; };
    int (*fp)(int) = L; 
    int (&fp2)(int) = [](auto a) { return a; };  // expected-error{{non-const lvalue}}
    int (&&fp3)(int) = [](auto a) { return a; };  // expected-error{{no viable conversion}}\
                                                  //expected-note{{candidate}}
  }
}

}


namespace return_type_deduction_ok {
 auto l = [](auto a) ->auto { return a; }(2); 
 auto l2 = [](auto a) ->decltype(auto) { return a; }(2);  
 auto l3 = [](auto a) { return a; }(2); 

}

namespace generic_lambda_as_default_argument_ok {
  void test(int i = [](auto a)->int { return a; }(3)) {
  }
}
