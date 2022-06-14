// RUN: %clang_cc1 -std=c++17 -fms-compatibility -fsyntax-only -verify=before,expected %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility -fdelayed-template-parsing -fsyntax-only -verify=before,expected %s
// RUN: %clang_cc1 -std=c++20 -fms-compatibility -fsyntax-only -verify=after,expected %s
// RUN: %clang_cc1 -std=c++20 -fms-compatibility -fdelayed-template-parsing -fsyntax-only -verify=after,expected %s

template <class T>
class Base {
};

template <class T>
class Based {}; // Trying to trick the typo detection

template <class T>
class Derived : public Base<T> {
public:
  // after-error@+1 {{member initializer 'Base' does not name a non-static data member or base class}}
  Derived() : Base() {} // before-warning {{unqualified base initializer of class templates is a Microsoft extension}}
private:
  int Baze; // Trying to trick the typo detection
};

template <class T> struct AggregateBase {
  T i;
};

template <class T>
struct AggregateDerived : public AggregateBase<T> {
  int i;

  // after-error@+1 {{member initializer 'AggregateBase' does not name a non-static data member or base class}}
  AggregateDerived(T j) : AggregateBase{4}, i{j} {} // before-warning {{unqualified base initializer of class templates is a Microsoft extension}}
  int f() {
    return i + AggregateBase::i; // expected-warning {{use of undeclared identifier 'AggregateBase'; unqualified lookup into dependent bases of class template 'AggregateDerived' is a Microsoft extension}}
  }
};

template <class T, typename U> struct MultiTypesBase {
};

template <class T, class U>
struct MultiTypesDerived : public MultiTypesBase<T, U> {
  // after-error@+1 {{member initializer 'MultiTypesBase' does not name a non-static data member or base class}}
  MultiTypesDerived() : MultiTypesBase{} {} // before-warning {{unqualified base initializer of class templates is a Microsoft extension}}
};

template <int I> struct IntegerBase {
};

template <int I>
struct IntegerDerived : public IntegerBase<I> {
  // after-error@+1 {{member initializer 'IntegerBase' does not name a non-static data member or base class}}
  IntegerDerived() : IntegerBase{} {} // before-warning {{unqualified base initializer of class templates is a Microsoft extension}}
};

template <class T> struct ConformingBase {
  T i;
};

template <class T>
struct ConformingDerived : public ConformingBase<T> {
  int i;

  ConformingDerived(T j) : ConformingBase<T>{4}, i{j} {}
  int f() {
    return i + ConformingBase<T>::i;
  }
};

int main() {
  int I;
  Derived<int> t;

  AggregateDerived<int> AD{2};
  AD.AggregateBase::i = 3;
  I = AD.f();

  MultiTypesDerived<int, double> MTD;

  IntegerDerived<4> ID;

  ConformingDerived<int> CD{2};
  I = CD.f();

  return I;
}
