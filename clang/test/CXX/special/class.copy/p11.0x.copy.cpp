// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct NonTrivial {
  NonTrivial(const NonTrivial&);
};

// A defaulted copy constructor for a class X is defined as deleted if X has:

// -- a variant member with a non-trivial corresponding constructor
union DeletedNTVariant { // expected-note{{here}}
  NonTrivial NT;
  DeletedNTVariant();
};
DeletedNTVariant DVa;
DeletedNTVariant DVb(DVa); // expected-error{{call to implicitly-deleted copy constructor}}

struct DeletedNTVariant2 { // expected-note{{here}}
  union {
    NonTrivial NT;
  };
  DeletedNTVariant2();
};
DeletedNTVariant2 DV2a;
DeletedNTVariant2 DV2b(DV2a); // expected-error{{call to implicitly-deleted copy constructor}}

// -- a non-static data member of class type M (or array thereof) that cannot be
//    copied because overload resolution results in an ambiguity or a function
//    that is deleted or inaccessible
struct NoAccess {
  NoAccess() = default;
private:
  NoAccess(const NoAccess&);

  friend struct HasAccess;
};

struct HasNoAccess { // expected-note{{here}}
  NoAccess NA;
};
HasNoAccess HNAa;
HasNoAccess HNAb(HNAa); // expected-error{{call to implicitly-deleted copy constructor}}

struct HasAccess {
  NoAccess NA;
};

HasAccess HAa;
HasAccess HAb(HAa);

struct NonConst {
  NonConst(NonConst&);
};
struct Ambiguity {
  Ambiguity(const Ambiguity&);
  Ambiguity(volatile Ambiguity&);
};

struct IsAmbiguous { // expected-note{{here}}
  NonConst NC;
  Ambiguity A;
  IsAmbiguous();
};
IsAmbiguous IAa;
IsAmbiguous IAb(IAa); // expected-error{{call to implicitly-deleted copy constructor}}

struct Deleted { // expected-note{{here}}
  IsAmbiguous IA;
};
Deleted Da;
Deleted Db(Da); // expected-error{{call to implicitly-deleted copy constructor}}

// -- a direct or virtual base class B that cannot be copied because overload
//    resolution results in an ambiguity or a function that is deleted or
//    inaccessible
struct AmbiguousCopyBase : Ambiguity { // expected-note {{here}}
  NonConst NC;
};
extern AmbiguousCopyBase ACBa;
AmbiguousCopyBase ACBb(ACBa); // expected-error {{deleted copy constructor}}

struct DeletedCopyBase : AmbiguousCopyBase {}; // expected-note {{here}}
extern DeletedCopyBase DCBa;
DeletedCopyBase DCBb(DCBa); // expected-error {{deleted copy constructor}}

struct InaccessibleCopyBase : NoAccess {}; // expected-note {{here}}
extern InaccessibleCopyBase ICBa;
InaccessibleCopyBase ICBb(ICBa); // expected-error {{deleted copy constructor}}

// -- any direct or virtual base class or non-static data member of a type with
//    a destructor that is deleted or inaccessible
struct NoAccessDtor {
private:
  ~NoAccessDtor();
  friend struct HasAccessDtor;
};

struct HasNoAccessDtor { // expected-note{{here}}
  NoAccessDtor NAD;
  HasNoAccessDtor();
  ~HasNoAccessDtor();
};
HasNoAccessDtor HNADa;
HasNoAccessDtor HNADb(HNADa); // expected-error{{call to implicitly-deleted copy constructor}}

struct HasAccessDtor {
  NoAccessDtor NAD;
};
HasAccessDtor HADa;
HasAccessDtor HADb(HADa);

struct HasNoAccessDtorBase : NoAccessDtor { // expected-note{{here}}
};
extern HasNoAccessDtorBase HNADBa;
HasNoAccessDtorBase HNADBb(HNADBa); // expected-error{{implicitly-deleted copy constructor}}

// -- a non-static data member of rvalue reference type
struct RValue { // expected-note{{here}}
  int && ri = 1;
};
RValue RVa;
RValue RVb(RVa); // expected-error{{call to implicitly-deleted copy constructor}}
