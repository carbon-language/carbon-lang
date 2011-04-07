// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

namespace std {
  class type_info {};
}

void one() { }
void two() { } // expected-note 3{{candidate}}
void two(int) { } // expected-note 3{{candidate}}

template<class T> void twoT() { } // expected-note 5{{candidate}}
template<class T> void twoT(int) { } // expected-note 5{{candidate}}

template<class T> void oneT() { }
template<class T, class U> void oneT(U) { }
/*
The target can be
 an object or reference being initialized (8.5, 8.5.3),
 the left side of an assignment (5.17),
 a parameter of a function (5.2.2),
 a parameter of a user-defined operator (13.5),
 the return value of a function, operator function, or conversion (6.6.3),
 an explicit type conversion (5.2.3, 5.2.9, 5.4), or
 a non-type template-parameter (14.3.2)
*/
//#include <typeinfo>
template<void (*p)(int)> struct test { };

int main()
{
   one;         // expected-warning {{expression result unused}}
   two;         // expected-error {{address of overloaded}}
   oneT<int>;  // expected-warning {{expression result unused}}
   twoT<int>;  // expected-error {{address of overloaded}}
   typeid(oneT<int>); // expected-warning{{expression result unused}}
  sizeof(oneT<int>); // expected-warning {{expression result unused}}
  sizeof(twoT<int>); //expected-error {{cannot resolve overloaded function 'twoT' from context}}
  decltype(oneT<int>)* fun = 0;
  
  *one;    // expected-warning {{expression result unused}}
  *oneT<int>;   // expected-warning {{expression result unused}}
  *two;  //expected-error {{cannot resolve overloaded function 'two' from context}}
  *twoT<int>; //expected-error {{cannot resolve overloaded function 'twoT' from context}}
  !oneT<int>;  // expected-warning {{expression result unused}}
  +oneT<int>;  // expected-warning {{expression result unused}}
  -oneT<int>;  //expected-error {{invalid argument type}}
  oneT<int> == 0;   // expected-warning {{expression result unused}}
  0 == oneT<int>;   // expected-warning {{expression result unused}}
  0 != oneT<int>;    // expected-warning {{expression result unused}}
  (false ? one : oneT<int>);   // expected-warning {{expression result unused}}
  void (*p1)(int); p1 = oneT<int>;
  
  int i = (int) (false ? (void (*)(int))twoT<int> : oneT<int>); //expected-error {{incompatible operand}}
  (twoT<int>) == oneT<int>; //expected-error {{cannot resolve overloaded function 'twoT' from context}}
  bool b = oneT<int>;
  void (*p)() = oneT<int>;
  test<oneT<int> > ti;
  void (*u)(int) = oneT<int>;

  b = (void (*)()) twoT<int>;
  
  one < one; //expected-warning {{self-comparison always evaluates to false}} \
             //expected-warning {{expression result unused}}         

  oneT<int> < oneT<int>;  //expected-warning {{self-comparison always evaluates to false}} \
                          //expected-warning {{expression result unused}}

  two < two; //expected-error {{cannot resolve overloaded function 'two' from context}}
  twoT<int> < twoT<int>; //expected-error {{cannot resolve overloaded function 'twoT' from context}}
  oneT<int> == 0;   // expected-warning {{expression result unused}}

}

struct rdar9108698 {
  template<typename> void f();
};

void test_rdar9108698(rdar9108698 x) {
  x.f<int>; // expected-error{{a bound member function may only be called}}
}
