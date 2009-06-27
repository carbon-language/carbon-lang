// RUN: clang-cc -fsyntax-only -verify %s

// Make sure we accept this
template<class X>struct A{typedef X Y;};
template<class X>bool operator==(A<X>,typename A<X>::Y);
int a(A<int> x) { return operator==(x,1); }

// FIXME: The diagnostic here is a bit messed up
template<class X>struct B{typedef X Y;};
template<class X>bool operator==(B<X>*,typename B<X>::Y); // \
expected-error{{overloaded 'operator==' must have at least one parameter of class or enumeration type}} \
expected-note{{in instantiation of default argument for 'operator==<int>' required here}}
int a(B<int> x) { return operator==(&x,1); }

