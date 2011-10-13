// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct NonTrivial {
  NonTrivial(const NonTrivial&);
};

union DeletedNTVariant { // expected-note{{here}}
  NonTrivial NT;
  DeletedNTVariant();
};
DeletedNTVariant DVa;
DeletedNTVariant DVb(DVa); // expected-error{{call to deleted constructor}}

struct DeletedNTVariant2 { // expected-note{{here}}
  union {
    NonTrivial NT;
  };
  DeletedNTVariant2();
};
DeletedNTVariant2 DV2a;
DeletedNTVariant2 DV2b(DV2a); // expected-error{{call to deleted constructor}}

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
HasNoAccess HNAb(HNAa); // expected-error{{call to deleted constructor}}

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
IsAmbiguous IAb(IAa); // expected-error{{call to deleted constructor}}

struct Deleted { // expected-note{{here}}
  IsAmbiguous IA;
};
Deleted Da;
Deleted Db(Da); // expected-error{{call to deleted constructor}}

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
HasNoAccessDtor HNADb(HNADa); // expected-error{{call to deleted constructor}}

struct HasAccessDtor {
  NoAccessDtor NAD;
};
HasAccessDtor HADa;
HasAccessDtor HADb(HADa);

struct RValue { // expected-note{{here}}
  int && ri = 1;
};
RValue RVa;
RValue RVb(RVa); // expected-error{{call to deleted constructor}}
