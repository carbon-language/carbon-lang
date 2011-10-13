// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct NonLiteral { NonLiteral(); };

// A type is a literal type if it is:

// - a scalar type
constexpr int f1(double);

// - a reference type
struct S { S(); };
constexpr int f2(S &);

// - a class type that has all of the following properties:

//  - it has a trivial destructor
struct UserProvDtor {
  constexpr UserProvDtor(); // expected-error {{non-literal type 'UserProvDtor' cannot have constexpr members}}
  ~UserProvDtor(); // expected-note {{has a user-provided destructor}}
};
struct NonTrivDtor {
  constexpr NonTrivDtor(); // expected-error {{non-literal type 'NonTrivDtor' cannot have constexpr members}}
  virtual ~NonTrivDtor() = default; // expected-note {{has a non-trivial destructor}}
};
struct NonTrivDtorBase {
  ~NonTrivDtorBase();
};
template<typename T>
struct DerivedFromNonTrivDtor : T { // expected-note {{'DerivedFromNonTrivDtor<NonTrivDtorBase>' is not literal because it has base class 'NonTrivDtorBase' of non-literal type}}
  constexpr DerivedFromNonTrivDtor();
};
constexpr int f(DerivedFromNonTrivDtor<NonTrivDtorBase>); // expected-error {{constexpr function's 1st parameter type 'DerivedFromNonTrivDtor<NonTrivDtorBase>' is not a literal type}}
struct TrivDtor {
  constexpr TrivDtor();
};
// FIXME: when building DefinitionData we look at 'isUserProvided' before it's set up!
#if 0
struct TrivDefaultedDtor {
  constexpr TrivDefaultedDtor();
  ~TrivDefaultedDtor() = default;
};
#endif

//  - it is an aggregate type or has at least one constexpr constructor or
//    constexpr constructor template that is not a copy or move constructor
struct Agg {
  int a;
  char *b;
};
constexpr int f3(Agg a) { return a.a; }
struct CtorTemplate {
  template<typename T> constexpr CtorTemplate(T);
};
struct CopyCtorOnly { // expected-note {{'CopyCtorOnly' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
  constexpr CopyCtorOnly(CopyCtorOnly&); // expected-error {{non-literal type 'CopyCtorOnly' cannot have constexpr members}}
};
struct MoveCtorOnly { // expected-note {{no constexpr constructors other than copy or move constructors}}
  constexpr MoveCtorOnly(MoveCtorOnly&&); // expected-error {{non-literal type 'MoveCtorOnly' cannot have constexpr members}}
};
template<typename T>
struct CtorArg { // expected-note {{no constexpr constructors other than copy or move constructors}}
  constexpr CtorArg(T); // expected-note {{constructor template instantiation is not constexpr because 1st parameter type 'NonLiteral' is not a literal type}}
};
constexpr int f(CtorArg<int>);
constexpr int f(CtorArg<NonLiteral>); // expected-error {{not a literal type}}
// We have a special-case diagnostic for classes with virtual base classes.
struct VBase {};
struct HasVBase : virtual VBase {}; // expected-note 2{{virtual base class declared here}}
struct Derived : HasVBase {
  constexpr Derived(); // expected-error {{constexpr constructor not allowed in struct with virtual base class}}
};
template<typename T> struct DerivedFromVBase : T { // expected-note {{struct with virtual base class is not a literal type}}
  constexpr DerivedFromVBase();
};
constexpr int f(DerivedFromVBase<HasVBase>); // expected-error {{constexpr function's 1st parameter type 'DerivedFromVBase<HasVBase>' is not a literal type}}

//  - it has all non-static data members and base classes of literal types
struct NonLitMember {
  S s; // expected-note {{has data member 's' of non-literal type 'S'}}
};
constexpr int f(NonLitMember); // expected-error {{1st parameter type 'NonLitMember' is not a literal type}}
struct NonLitBase :
  S { // expected-note {{base class 'S' of non-literal type}}
  constexpr NonLitBase(); // expected-error {{non-literal type 'NonLitBase' cannot have constexpr members}}
};
struct LitMemBase : Agg {
  Agg agg;
};
template<typename T>
struct MemberType {
  T t; // expected-note {{'MemberType<NonLiteral>' is not literal because it has data member 't' of non-literal type 'NonLiteral'}}
  constexpr MemberType();
};
constexpr int f(MemberType<int>);
constexpr int f(MemberType<NonLiteral>); // expected-error {{not a literal type}}

// - an array of literal type
struct ArrGood {
  Agg agg[24];
  double d[12];
  TrivDtor td[3];
};
constexpr int f(ArrGood);

struct ArrBad {
  S s[3]; // expected-note {{data member 's' of non-literal type 'S [3]'}}
};
constexpr int f(ArrBad); // expected-error {{1st parameter type 'ArrBad' is not a literal type}}


// As a non-conforming tweak to the standard, we do not allow a literal type to
// have any mutable data members.
namespace MutableMembers {
  struct MM {
    mutable int n; // expected-note {{'MM' is not literal because it has a mutable data member}}
  };
  constexpr int f(MM); // expected-error {{not a literal type}}

  // Here's one reason why allowing this would be a disaster...
  template<int n> struct Id { int k = n; };
  int f() {
    // FIXME: correctly check whether the initializer is a constant expression.
    constexpr MM m = { 0 }; // desired-error {{must be a constant expression}}
    ++m.n;
    return Id<m.n>().k; // expected-error {{not an integral constant expression}}
  }
}
