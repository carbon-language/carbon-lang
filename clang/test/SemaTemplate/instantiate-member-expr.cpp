// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct S {
 S() { }
};

template<typename T>
struct vector {
  void push_back(const T&) { int a[sizeof(T) ? -1: -1]; } // expected-error {{array size is negative}}
};

class GRExprEngine {
public:
 typedef vector<S<void *> >CheckersOrdered;
 CheckersOrdered Checkers;

 template <typename CHECKER>
 void registerCheck(CHECKER *check) {
   Checkers.push_back(S<void *>()); // expected-note {{in instantiation of member function 'vector<struct S<void *> >::push_back' requested here}}
 }
};

class RetainReleaseChecker { };

void f(GRExprEngine& Eng) {
   Eng.registerCheck(new RetainReleaseChecker); // expected-note {{in instantiation of function template specialization 'GRExprEngine::registerCheck<class RetainReleaseChecker>' requested here}}
}

// PR 5838
namespace test1 {
  template<typename T> struct A {
    int a;
  };

  template<typename T> struct B : A<float>, A<T> {
    void f() {
      a = 0; // should not be ambiguous
    }
  };
  template struct B<int>;

  struct O {
    int a;
    template<typename T> struct B : A<T> {
      void f() {
        a = 0; // expected-error {{type 'struct test1::O' is not a direct or virtual base of ''B<int>''}}
      }
    };
  };
  template struct O::B<int>; // expected-note {{in instantiation}}
}
