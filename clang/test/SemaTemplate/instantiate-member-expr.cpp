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
