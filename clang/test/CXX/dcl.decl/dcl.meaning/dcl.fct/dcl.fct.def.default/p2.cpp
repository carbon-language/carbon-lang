// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// FIXME: test with non-std qualifiers

namespace move {
  struct Const {
    Const(const Const&&) = default; // expected-error {{the parameter for an explicitly-defaulted move constructor may not be const}}
    Const& operator=(const Const&&) = default; // expected-error {{the parameter for an explicitly-defaulted move assignment operator may not be const}}
  };

  struct Volatile {
    Volatile(volatile Volatile&&) = default; // expected-error {{the parameter for an explicitly-defaulted move constructor may not be volatile}}
    Volatile& operator=(volatile Volatile&&) = default; // expected-error {{the parameter for an explicitly-defaulted move assignment operator may not be volatile}}
  };

  struct AssignmentRet1 {
    AssignmentRet1&& operator=(AssignmentRet1&&) = default; // expected-error {{an explicitly-defaulted move assignment operator must return an unqualified lvalue reference to its class type}}
  };

  struct AssignmentRet2 {
    const AssignmentRet2& operator=(AssignmentRet2&&) = default; // expected-error {{an explicitly-defaulted move assignment operator must return an unqualified lvalue reference to its class type}}
  };

  struct ConstAssignment {
    ConstAssignment& operator=(ConstAssignment&&) const = default; // expected-error {{an explicitly-defaulted move assignment operator may not have 'const' or 'volatile' qualifiers}}
  };
}

namespace copy {
  struct Volatile {
    Volatile(const volatile Volatile&) = default; // expected-error {{the parameter for an explicitly-defaulted copy constructor may not be volatile}}
    Volatile& operator=(const volatile Volatile&) = default; // expected-error {{the parameter for an explicitly-defaulted copy assignment operator may not be volatile}}
  };

  struct Const {
    Const(const Const&) = default;
    Const& operator=(const Const&) = default;
  };

  struct NonConst {
    NonConst(NonConst&) = default;
    NonConst& operator=(NonConst&) = default;
  };

  struct BadConst {
    NonConst nc; // makes implicit copy non-const
    BadConst(const BadConst&) = default; // expected-error {{is const, but}}
    BadConst& operator=(const BadConst&) = default; // expected-error {{is const, but}}
  };

  struct AssignmentRet1 {
    AssignmentRet1&& operator=(const AssignmentRet1&) = default; // expected-error {{an explicitly-defaulted copy assignment operator must return an unqualified lvalue reference to its class type}}
  };

  struct AssignmentRet2 {
    const AssignmentRet2& operator=(const AssignmentRet2&) = default; // expected-error {{an explicitly-defaulted copy assignment operator must return an unqualified lvalue reference to its class type}}
  };

  struct ConstAssignment {
    ConstAssignment& operator=(const ConstAssignment&) const = default; // expected-error {{an explicitly-defaulted copy assignment operator may not have 'const' or 'volatile' qualifiers}}
  };
}
