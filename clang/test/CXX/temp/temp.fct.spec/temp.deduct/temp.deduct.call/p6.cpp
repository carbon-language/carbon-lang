// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  template<class T> void apply(T x, void (*f)(T)) { f(x); } // expected-note 2 {{failed template argument deduction}}\
  // expected-note {{no overload of 'temp2' matching 'void (*)(int)'}}

  template<class A> void temp(A);
  void test0() {
    // okay: deduce T=int from first argument, A=int during overload
    apply(0, &temp);
    apply(0, &temp<>);

    // okay: deduce T=int from first and second arguments
    apply(0, &temp<int>);

    // deduction failure: T=int from first, T=long from second
    apply(0, &temp<long>); // expected-error {{no matching function for call to 'apply'}}
  }

  void over(int);
  int over(long);

  void test1() {
    // okay: deductions match
    apply(0, &over);

    // deduction failure: deduced T=long from first argument, T=int from second
    apply(0L, &over); // expected-error {{no matching function for call to 'apply'}}
  }

  void over(short);

  void test2() {
    // deduce T=int from first arg, second arg is undeduced context,
    // pick correct overload of 'over' during overload resolution for 'apply'
    apply(0, &over);
  }

  template<class A, class B> B temp2(A);
  void test3() {
    // deduce T=int from first arg, A=int B=void during overload resolution
    apply(0, &temp2);
    apply(0, &temp2<>);
    apply(0, &temp2<int>);

    // overload failure
    apply(0, &temp2<long>); // expected-error {{no matching function for call to 'apply'}}
  }
}

namespace test1 {
  template<class T> void invoke(void (*f)(T)) { f(T()); } // expected-note 6 {{couldn't infer template argument}} \
  // expected-note {{failed template argument deduction}}

  template<class T> void temp(T);
  void test0() {
    // deduction failure: overload has template => undeduced context
    invoke(&temp); // expected-error {{no matching function for call to 'invoke'}}
    invoke(&temp<>); // expected-error {{no matching function for call to 'invoke'}}

    // okay: full template-id
    invoke(&temp<int>);
  }

  void over(int);
  int over(long);

  void test1() {
    // okay: only one overload matches
    invoke(&over);
  }

  void over(short);

  void test2() {
    // deduction failure: overload has multiple matches => undeduced context
    invoke(&over); // expected-error {{no matching function for call to 'invoke'}}
  }

  template<class A, class B> B temp2(A);
  void test3() {
    // deduction failure: overload has template => undeduced context
    // (even though partial application temp2<int> could in theory
    // let us infer T=int)
    invoke(&temp2); // expected-error {{no matching function for call to 'invoke'}}
    invoke(&temp2<>); // expected-error {{no matching function for call to 'invoke'}}
    invoke(&temp2<int>); // expected-error {{no matching function for call to 'invoke'}}

    // okay: full template-id
    invoke(&temp2<int, void>);

    // overload failure
    invoke(&temp2<int, int>); // expected-error {{no matching function for call to 'invoke'}}
  }
}
