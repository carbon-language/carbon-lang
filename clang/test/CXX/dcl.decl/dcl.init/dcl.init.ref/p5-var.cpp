// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Base { }; // expected-note{{candidate function}}
struct Derived : Base { }; // expected-note{{candidate function}}
struct Unrelated { };
struct Derived2 : Base { };
struct Diamond : Derived, Derived2 { };

struct ConvertibleToBaseRef {
  operator Base&() const;
};

struct ConvertibleToDerivedRef {
  operator Derived&() const;
};

struct ConvertibleToBothDerivedRef {
  operator Derived&(); // expected-note{{candidate function}}
  operator Derived2&(); // expected-note{{candidate function}}
};

struct ConvertibleToIntRef {
  operator int&();
};

struct ConvertibleToBase {
  operator Base() const;
};

struct ConvertibleToDerived {
  operator Derived() const;
};

struct ConvertibleToBothDerived {
  operator Derived(); // expected-note{{candidate function}}
  operator Derived2(); // expected-note{{candidate function}}
};

struct ConvertibleToInt {
  operator int();
};

template<typename T> T create();

// First bullet: lvalue references binding to lvalues (the simple cases).
void bind_lvalue_to_lvalue(Base b, Derived d, 
                           const Base bc, const Derived dc,
                           Diamond diamond,
                           int i) {
  // Reference-compatible
  Base &br1 = b;
  Base &br2 = d;
  Derived &dr1 = d;
  Derived &dr2 = b; // expected-error{{non-const lvalue reference to type 'struct Derived' cannot bind to a value of unrelated type 'struct Base'}}
  Base &br3 = bc; // expected-error{{drops qualifiers}}
  Base &br4 = dc; // expected-error{{drops qualifiers}}
  Base &br5 = diamond; // expected-error{{ambiguous conversion from derived class 'struct Diamond' to base class 'struct Base'}}
  int &ir = i;
  long &lr = i; // expected-error{{non-const lvalue reference to type 'long' cannot bind to a value of unrelated type 'int'}}
}

void bind_lvalue_quals(volatile Base b, volatile Derived d,
                       volatile const Base bvc, volatile const Derived dvc,
                       volatile const int ivc) {
  volatile Base &bvr1 = b;
  volatile Base &bvr2 = d;
  volatile Base &bvr3 = bvc; // expected-error{{binding of reference to type 'struct Base volatile' to a value of type 'struct Base const volatile' drops qualifiers}}
  volatile Base &bvr4 = dvc; // expected-error{{binding of reference to type 'struct Base volatile' to a value of type 'struct Derived const volatile' drops qualifiers}}
  
  volatile int &ir = ivc; // expected-error{{binding of reference to type 'int volatile' to a value of type 'int const volatile' drops qualifiers}}
}

void bind_lvalue_to_rvalue() {
  Base &br1 = Base(); // expected-error{{non-const lvalue reference to type 'struct Base' cannot bind to a temporary of type 'struct Base'}}
  Base &br2 = Derived(); // expected-error{{non-const lvalue reference to type 'struct Base' cannot bind to a temporary of type 'struct Derived'}}

  int &ir = 17; // expected-error{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
}

void bind_lvalue_to_unrelated(Unrelated ur) {
  Base &br1 = ur; // expected-error{{non-const lvalue reference to type 'struct Base' cannot bind to a value of unrelated type 'struct Unrelated'}}
}

void bind_lvalue_to_conv_lvalue() {
  // Not reference-related, but convertible
  Base &nbr1 = ConvertibleToBaseRef();
  Base &nbr2 = ConvertibleToDerivedRef();
  Derived &ndr1 = ConvertibleToDerivedRef();
  int &ir = ConvertibleToIntRef();
}

void bind_lvalue_to_conv_lvalue_ambig(ConvertibleToBothDerivedRef both) {
  Derived &dr1 = both;
  Base &br1 = both; // expected-error{{reference initialization of type 'struct Base &' with initializer of type 'struct ConvertibleToBothDerivedRef' is ambiguous}}
}

struct IntBitfield {
  int i : 17; // expected-note{{bit-field is declared here}}
};

void test_bitfield(IntBitfield ib) {
  int & ir1 = (ib.i); // expected-error{{non-const reference cannot bind to bit-field 'i'}}
}

// Second bullet: const lvalue reference binding to an rvalue with
// similar type (both of which are class types).
void bind_const_lvalue_to_rvalue() {
  const Base &br1 = create<Base>();
  const Base &br2 = create<Derived>();
  const Derived &dr1 = create<Base>(); // expected-error{{no viable conversion}}

  const Base &br3 = create<const Base>();
  const Base &br4 = create<const Derived>();

  const Base &br5 = create<const volatile Base>(); // expected-error{{binding of reference to type 'struct Base const' to a value of type 'struct Base const volatile' drops qualifiers}}
  const Base &br6 = create<const volatile Derived>(); // expected-error{{binding of reference to type 'struct Base const' to a value of type 'struct Derived const volatile' drops qualifiers}}

  const int &ir = create<int>();
}

// Second bullet: const lvalue reference binds to the result of a conversion.
void bind_const_lvalue_to_class_conv_temporary() {
  const Base &br1 = ConvertibleToBase();
  const Base &br2 = ConvertibleToDerived();
}
void bind_lvalue_to_conv_rvalue_ambig(ConvertibleToBothDerived both) {
  const Derived &dr1 = both;
  const Base &br1 = both; // expected-error{{reference initialization of type 'struct Base const &' with initializer of type 'struct ConvertibleToBothDerived' is ambiguous}}
}
