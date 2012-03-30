// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Tests for implicit (non-)declaration of move constructor and
// assignment: p9, p11, p20, p23.

// This class, used as a member, allows to distinguish move from copy because
// move operations are no-throw, copy operations aren't.
struct ThrowingCopy {
  ThrowingCopy() noexcept;
  ThrowingCopy(ThrowingCopy &&) noexcept;
  ThrowingCopy(const ThrowingCopy &) noexcept(false);
  ThrowingCopy & operator =(ThrowingCopy &&) noexcept;
  ThrowingCopy & operator =(const ThrowingCopy &) noexcept(false);
};

struct HasCopyConstructor {
  ThrowingCopy tc;
  HasCopyConstructor() noexcept;
  HasCopyConstructor(const HasCopyConstructor &) noexcept(false);
};

struct HasCopyAssignment {
  ThrowingCopy tc;
  HasCopyAssignment() noexcept;
  HasCopyAssignment & operator =(const HasCopyAssignment &) noexcept(false);
};

struct HasMoveConstructor { // expected-note {{implicit copy assignment}}
  ThrowingCopy tc;
  HasMoveConstructor() noexcept;
  HasMoveConstructor(HasMoveConstructor &&) noexcept;
};

struct HasMoveAssignment { // expected-note {{implicit copy constructor}}
  ThrowingCopy tc;
  HasMoveAssignment() noexcept;
  HasMoveAssignment & operator =(HasMoveAssignment &&) noexcept;
};

struct HasDestructor {
  ThrowingCopy tc;
  HasDestructor() noexcept;
  ~HasDestructor() noexcept;
};

void test_basic_exclusion() {
  static_assert(!noexcept(HasCopyConstructor((HasCopyConstructor()))), "");
  HasCopyConstructor hcc;
  static_assert(!noexcept(hcc = HasCopyConstructor()), "");

  static_assert(!noexcept(HasCopyAssignment((HasCopyAssignment()))), "");
  HasCopyAssignment hca;
  static_assert(!noexcept(hca = HasCopyAssignment()), "");

  static_assert(noexcept(HasMoveConstructor((HasMoveConstructor()))), "");
  HasMoveConstructor hmc;
  hmc = HasMoveConstructor(); // expected-error {{selected implicitly-deleted copy assignment}}

  (HasMoveAssignment(HasMoveAssignment())); // expected-error {{uses deleted function}}
  HasMoveAssignment hma;
  static_assert(noexcept(hma = HasMoveAssignment()), "");

  static_assert(!noexcept(HasDestructor((HasDestructor()))), "");
  HasDestructor hd;
  static_assert(!noexcept(hd = HasDestructor()), "");
}

struct PrivateMove {
  PrivateMove() noexcept;
  PrivateMove(const PrivateMove &) noexcept(false);
  PrivateMove & operator =(const PrivateMove &) noexcept(false);
private:
  PrivateMove(PrivateMove &&) noexcept;
  PrivateMove & operator =(PrivateMove &&) noexcept;
};

struct InheritsPrivateMove : PrivateMove {};
struct ContainsPrivateMove {
  PrivateMove pm;
};

struct PrivateDestructor {
  PrivateDestructor() noexcept;
  PrivateDestructor(const PrivateDestructor &) noexcept(false);
  PrivateDestructor(PrivateDestructor &&) noexcept;
private:
  ~PrivateDestructor() noexcept;
};

struct InheritsPrivateDestructor : PrivateDestructor {}; // expected-note{{base class 'PrivateDestructor' has an inaccessible destructor}}
struct ContainsPrivateDestructor {
  PrivateDestructor pd; // expected-note{{field 'pd' has an inaccessible destructor}}
};

struct NonTrivialCopyOnly {
  NonTrivialCopyOnly() noexcept;
  NonTrivialCopyOnly(const NonTrivialCopyOnly &) noexcept(false);
  NonTrivialCopyOnly & operator =(const NonTrivialCopyOnly &) noexcept(false);
};

struct InheritsNonTrivialCopyOnly : NonTrivialCopyOnly {};
struct ContainsNonTrivialCopyOnly {
  NonTrivialCopyOnly ntco;
};

struct ContainsConst {
  const int i;
  ContainsConst() noexcept;
  ContainsConst & operator =(ContainsConst &); // expected-note {{not viable}}
};

struct ContainsRef {
  int &i;
  ContainsRef() noexcept;
  ContainsRef & operator =(ContainsRef &); // expected-note {{not viable}}
};

struct Base {
  Base & operator =(Base &);
};
struct DirectVirtualBase : virtual Base {}; // expected-note {{copy assignment operator) not viable}}
struct IndirectVirtualBase : DirectVirtualBase {}; // expected-note {{copy assignment operator) not viable}}

void test_deletion_exclusion() {
  // FIXME: How to test the union thing?

  static_assert(!noexcept(InheritsPrivateMove(InheritsPrivateMove())), "");
  static_assert(!noexcept(ContainsPrivateMove(ContainsPrivateMove())), "");
  InheritsPrivateMove ipm;
  static_assert(!noexcept(ipm = InheritsPrivateMove()), "");
  ContainsPrivateMove cpm;
  static_assert(!noexcept(cpm = ContainsPrivateMove()), "");

  (InheritsPrivateDestructor(InheritsPrivateDestructor())); // expected-error {{call to implicitly-deleted default constructor}}
  (ContainsPrivateDestructor(ContainsPrivateDestructor())); // expected-error {{call to implicitly-deleted default constructor}}

  static_assert(!noexcept(InheritsNonTrivialCopyOnly(InheritsNonTrivialCopyOnly())), "");
  static_assert(!noexcept(ContainsNonTrivialCopyOnly(ContainsNonTrivialCopyOnly())), "");
  InheritsNonTrivialCopyOnly intco;
  static_assert(!noexcept(intco = InheritsNonTrivialCopyOnly()), "");
  ContainsNonTrivialCopyOnly cntco;
  static_assert(!noexcept(cntco = ContainsNonTrivialCopyOnly()), "");

  ContainsConst cc;
  cc = ContainsConst(); // expected-error {{no viable}} 

  ContainsRef cr;
  cr = ContainsRef(); // expected-error {{no viable}} 

  DirectVirtualBase dvb;
  dvb = DirectVirtualBase(); // expected-error {{no viable}} 

  IndirectVirtualBase ivb;
  ivb = IndirectVirtualBase(); // expected-error {{no viable}} 
}

struct ContainsRValueRef {
  int&& ri;
  ContainsRValueRef() noexcept;
};

void test_contains_rref() {
  (ContainsRValueRef(ContainsRValueRef()));
}
