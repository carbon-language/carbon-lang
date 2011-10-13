// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct NonTrivial {
  NonTrivial(NonTrivial&&);
};

union DeletedNTVariant {
  NonTrivial NT;
  DeletedNTVariant(DeletedNTVariant&&);
};
DeletedNTVariant::DeletedNTVariant(DeletedNTVariant&&) = default; // expected-error{{would delete}}

struct DeletedNTVariant2 {
  union {
    NonTrivial NT;
  };
  DeletedNTVariant2(DeletedNTVariant2&&);
};
DeletedNTVariant2::DeletedNTVariant2(DeletedNTVariant2&&) = default; // expected-error{{would delete}}

struct NoAccess {
  NoAccess() = default;
private:
  NoAccess(NoAccess&&);

  friend struct HasAccess;
};

struct HasNoAccess {
  NoAccess NA;
  HasNoAccess(HasNoAccess&&);
};
HasNoAccess::HasNoAccess(HasNoAccess&&) = default; // expected-error{{would delete}}

struct HasAccess {
  NoAccess NA;
  HasAccess(HasAccess&&);
};
HasAccess::HasAccess(HasAccess&&) = default;

struct NoAccessDtor {
  NoAccessDtor(NoAccessDtor&&);
private:
  ~NoAccessDtor();
  friend struct HasAccessDtor;
};

struct HasNoAccessDtor {
  NoAccessDtor NAD;
  HasNoAccessDtor(HasNoAccessDtor&&);
};
HasNoAccessDtor::HasNoAccessDtor(HasNoAccessDtor&&) = default; // expected-error{{would delete}}

struct HasAccessDtor {
  NoAccessDtor NAD;
  HasAccessDtor(HasAccessDtor&&);
};
HasAccessDtor::HasAccessDtor(HasAccessDtor&&) = default;

struct RValue {
  int &&ri = 1;
  RValue(RValue&&);
};
RValue::RValue(RValue&&) = default;

struct CopyOnly {
  CopyOnly(const CopyOnly&);
};

struct NonMove {
  CopyOnly CO;
  NonMove(NonMove&&);
};
NonMove::NonMove(NonMove&&) = default; // expected-error{{would delete}}

struct Moveable {
  Moveable();
  Moveable(Moveable&&);
};

struct HasMove {
  Moveable M;
  HasMove(HasMove&&);
};
HasMove::HasMove(HasMove&&) = default;
