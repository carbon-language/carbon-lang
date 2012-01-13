// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++11-extensions %s

struct errc {
  int v_;
  operator int() const {return v_;}
};

class error_condition
{
  int _val_;
public:
  error_condition() : _val_(0) {}

  error_condition(int _val)
    : _val_(_val) {}

  template <class E>
  error_condition(E _e) {
    // make_error_condition must not be typo corrected to error_condition
    // even though the first declaration of make_error_condition has not
    // yet been encountered. This was a bug in the first version of the type
    // name typo correction patch that wasn't noticed until building LLVM with
    // Clang failed.
    *this = make_error_condition(_e);
  }

};

inline error_condition make_error_condition(errc _e) {
  return error_condition(static_cast<int>(_e));
}


// Prior to the introduction of a callback object to further filter possible
// typo corrections, this example would not trigger a suggestion as "base_type"
// is a closer match to "basetype" than is "BaseType" but "base_type" does not
// refer to a base class or non-static data member.
struct BaseType { };
struct Derived : public BaseType { // expected-note {{base class 'BaseType' specified here}}
  static int base_type; // expected-note {{'base_type' declared here}}
  Derived() : basetype() {} // expected-error{{initializer 'basetype' does not name a non-static data member or base class; did you mean the base class 'BaseType'?}}
};

int get_type(struct Derived *st) {
  return st->Base_Type; // expected-error{{no member named 'Base_Type' in 'Derived'; did you mean 'base_type'?}}
}

// In this example, somename should not be corrected to the cached correction
// "some_name" since "some_name" is a class and a namespace name is needed.
class some_name {}; // expected-note {{'some_name' declared here}}
somename Foo; // expected-error {{unknown type name 'somename'; did you mean 'some_name'?}}
namespace SomeName {} // expected-note {{namespace 'SomeName' defined here}}
using namespace somename; // expected-error {{no namespace named 'somename'; did you mean 'SomeName'?}}


// Without the callback object, CorrectTypo would choose "field1" as the
// correction for "fielda" as it is closer than "FieldA", but that correction
// would be later discarded by the caller and no suggestion would be given.
struct st {
  struct {
    int field1;
  };
  double FieldA; // expected-note{{'FieldA' declared here}}
};
st var = { .fielda = 0.0 }; // expected-error{{field designator 'fielda' does not refer to any field in type 'st'; did you mean 'FieldA'?}}

// Test the improvement from passing a  callback object to CorrectTypo in
// Sema::BuildCXXNestedNameSpecifier.
typedef char* another_str;
namespace AnotherStd { // expected-note{{'AnotherStd' declared here}}
  class string {};
}
another_std::string str; // expected-error{{use of undeclared identifier 'another_std'; did you mean 'AnotherStd'?}}
