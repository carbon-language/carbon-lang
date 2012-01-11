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
  static int base_type;
  Derived() : basetype() {} // expected-error{{initializer 'basetype' does not name a non-static data member or base class; did you mean the base class 'BaseType'?}}
};

// In this example, somename should not be corrected to the cached correction
// "some_name" since "some_name" is a class and a namespace name is needed.
class some_name {}; // expected-note {{'some_name' declared here}}
somename Foo; // expected-error {{unknown type name 'somename'; did you mean 'some_name'?}}
namespace SomeName {} // expected-note {{namespace 'SomeName' defined here}}
using namespace somename; // expected-error {{no namespace named 'somename'; did you mean 'SomeName'?}}
