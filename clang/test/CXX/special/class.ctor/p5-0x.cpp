// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct DefaultedDefCtor1 {};
struct DefaultedDefCtor2 { DefaultedDefCtor2() = default; };
struct DeletedDefCtor { DeletedDefCtor() = delete; DeletedDefCtor(int); };
class PrivateDefCtor { PrivateDefCtor() = default; public: PrivateDefCtor(int); };
struct DeletedDtor { ~DeletedDtor() = delete; };
class PrivateDtor { ~PrivateDtor() = default; };
class Friend {
  Friend() = default; ~Friend() = default;
  friend struct NotDeleted6c;
  friend struct NotDeleted7i;
  friend struct NotDeleted7j;
  friend struct NotDeleted7k;
};
struct UserProvidedDefCtor { UserProvidedDefCtor() {} };
int n;


// A defaulted default constructor for a class X is defined as deleted if:

// - X is a union-like class that has a variant member with a non-trivial
// default constructor,
union Deleted1a { UserProvidedDefCtor u; }; // expected-note {{deleted here}}
Deleted1a d1a; // expected-error {{deleted constructor}}
union NotDeleted1a { DefaultedDefCtor1 nu; };
NotDeleted1a nd1a;
// FIXME: clang implements the pre-FDIS rule, under which DefaultedDefCtor2's
// default constructor is non-trivial.
union NotDeleted1b { DefaultedDefCtor2 nu; }; // unexpected-note {{deleted here}}
NotDeleted1b nd1b; // unexpected-error {{deleted constructor}}

// - any non-static data member with no brace-or-equal-initializer is of
// reference type,
class Deleted2a { Deleted2a() = default; int &a; }; // expected-note {{deleted here}}
Deleted2a d2a; // expected-error {{deleted constructor}}
class NotDeleted2a { int &a = n; };
NotDeleted2a nd2a;
class NotDeleted2b { int &a = error; }; // expected-error {{undeclared identifier}}
NotDeleted2b nd2b;

// - any non-variant non-static data member of const qualified type (or array
// thereof) with no brace-or-equal-initializer does not have a user-provided
// default constructor,
class Deleted3a { const int a; }; // expected-note {{here}} \
                                     expected-warning {{does not declare any constructor}} \
                                     expected-note {{will never be initialized}}
Deleted3a d3a; // expected-error {{deleted constructor}}
class Deleted3b { const DefaultedDefCtor1 a[42]; }; // expected-note {{here}}
Deleted3b d3b; // expected-error {{deleted constructor}}
// FIXME: clang implements the pre-FDIS rule, under which DefaultedDefCtor2's
// default constructor is user-provided.
class Deleted3c { const DefaultedDefCtor2 a; }; // desired-note {{here}}
Deleted3c d3c; // desired-error {{deleted constructor}}
class NotDeleted3a { const int a = 0; };
NotDeleted3a nd3a;
class NotDeleted3b { const DefaultedDefCtor1 a[42] = {}; };
NotDeleted3b nd3b;
class NotDeleted3c { const DefaultedDefCtor2 a = DefaultedDefCtor2(); };
NotDeleted3c nd3c;
union NotDeleted3d { const int a; int b; };
NotDeleted3d nd3d;
// FIXME: this class should not have a deleted default constructor.
union NotDeleted3e { const DefaultedDefCtor1 a[42]; int b; }; // unexpected-note {{here}}
NotDeleted3e nd3e; // unexpected-error {{deleted constructor}}
// FIXME: clang implements the pre-FDIS rule, under which DefaultedDefCtor2 is
// non-trivial.
union NotDeleted3f { const DefaultedDefCtor2 a; int b; }; // unexpected-note {{here}}
NotDeleted3f nd3f; // unexpected-error {{deleted constructor}}

// - X is a union and all of its variant members are of const-qualified type (or
// array thereof),
union Deleted4a { const int a; const int b; const UserProvidedDefCtor c; }; // expected-note {{here}}
Deleted4a d4a; // expected-error {{deleted constructor}}
union Deleted4b { const int a; int b; };
Deleted4b d4b;

// - X is a non-union class and all members of any anonymous union member are of
// const-qualified type (or array thereof),
struct Deleted5a { union { const int a; }; union { int b; }; }; // expected-note {{here}}
Deleted5a d5a; // expected-error {{deleted constructor}}
struct Deleted5b { union { const int a; int b; }; union { const int c; int d; }; };
Deleted5b d5b;

// - any direct or virtual base class, or non-static data member with no
// brace-or-equal-initializer, has class type M (or array thereof) and either
// M has no default constructor or overload resolution as applied to M's default
// constructor results in an ambiguity or in a function that is deleted or
// inaccessible from the defaulted default constructor, or
struct Deleted6a : Deleted2a {}; // expected-note {{here}}
Deleted6a d6a; // expected-error {{deleted constructor}}
struct Deleted6b : virtual Deleted2a {}; // expected-note {{here}}
Deleted6b d6b; // expected-error {{deleted constructor}}
struct Deleted6c { Deleted2a a; }; // expected-note {{here}}
Deleted6c d6c; // expected-error {{deleted constructor}}
struct Deleted6d { DeletedDefCtor a; }; // expected-note {{here}}
Deleted6d d6d; // expected-error {{deleted constructor}}
struct NotDeleted6a { DeletedDefCtor a = 0; };
NotDeleted6a nd6a;
struct Deleted6e { PrivateDefCtor a; }; // expected-note {{here}}
Deleted6e d6e; // expected-error {{deleted constructor}}
struct NotDeleted6b { PrivateDefCtor a = 0; };
NotDeleted6b nd6b;
struct NotDeleted6c { Friend a; };
NotDeleted6c nd6c;

// - any direct or virtual base class or non-static data member has a type with
// a destructor that is deleted or inaccessible from the defaulted default
// constructor.
struct Deleted7a : DeletedDtor {}; // expected-note {{here}}
Deleted7a d7a; // expected-error {{deleted constructor}}
struct Deleted7b : virtual DeletedDtor {}; // expected-note {{here}}
Deleted7b d7b; // expected-error {{deleted constructor}}
struct Deleted7c { DeletedDtor a; }; // expected-note {{here}}
Deleted7c d7c; // expected-error {{deleted constructor}}
struct Deleted7d { DeletedDtor a = {}; }; // expected-note {{here}}
Deleted7d d7d; // expected-error {{deleted constructor}}
struct Deleted7e : PrivateDtor {}; // expected-note {{here}}
Deleted7e d7e; // expected-error {{deleted constructor}}
struct Deleted7f : virtual PrivateDtor {}; // expected-note {{here}}
Deleted7f d7f; // expected-error {{deleted constructor}}
struct Deleted7g { PrivateDtor a; }; // expected-note {{here}}
Deleted7g d7g; // expected-error {{deleted constructor}}
struct Deleted7h { PrivateDtor a = {}; }; // expected-note {{here}}
Deleted7h d7h; // expected-error {{deleted constructor}}
struct NotDeleted7i : Friend {};
NotDeleted7i d7i;
struct NotDeleted7j : virtual Friend {};
NotDeleted7j d7j;
struct NotDeleted7k { Friend a; };
NotDeleted7k d7k;


class Trivial { static const int n = 42; };
static_assert(__has_trivial_constructor(Trivial), "Trivial is nontrivial");

// A default constructor is trivial if it is not user-provided and if:
class NonTrivialDefCtor1 { NonTrivialDefCtor1(); };
static_assert(!__has_trivial_constructor(NonTrivialDefCtor1), "NonTrivialDefCtor1 is trivial");

// - its class has no virtual functions (10.3) and no virtual base classes (10.1), and
class NonTrivialDefCtor2 { virtual void f(); };
static_assert(!__has_trivial_constructor(NonTrivialDefCtor2), "NonTrivialDefCtor2 is trivial");
class NonTrivialDefCtor3 : virtual Trivial {};
static_assert(!__has_trivial_constructor(NonTrivialDefCtor3), "NonTrivialDefCtor3 is trivial");

// - no non-static data member of its class has a brace-or-equal-initializer, and
class NonTrivialDefCtor4 { int m = 52; };
static_assert(!__has_trivial_constructor(NonTrivialDefCtor4), "NonTrivialDefCtor4 is trivial");

// - all the direct base classes of its class have trivial default constructors, and
class NonTrivialDefCtor5 : NonTrivialDefCtor1 {};
static_assert(!__has_trivial_constructor(NonTrivialDefCtor5), "NonTrivialDefCtor5 is trivial");

// - for all the non-static data members of its class that are of class type (or array thereof), each such class
// has a trivial default constructor.
class NonTrivialDefCtor6 { NonTrivialDefCtor1 t; };
static_assert(!__has_trivial_constructor(NonTrivialDefCtor6), "NonTrivialDefCtor5 is trivial");

// Otherwise, the default constructor is non-trivial.
class Trivial2 { Trivial2() = delete; };
//static_assert(__has_trivial_constructor(Trivial2), "NonTrivialDefCtor2 is trivial");
// FIXME: clang implements the pre-FDIS rule, under which this class is non-trivial.
static_assert(!__has_trivial_constructor(Trivial2), "NonTrivialDefCtor2 is trivial");

class Trivial3 { Trivial3() = default; };
//static_assert(__has_trivial_constructor(Trivial3), "NonTrivialDefCtor3 is trivial");
// FIXME: clang implements the pre-FDIS rule, under which this class is non-trivial.
static_assert(!__has_trivial_constructor(Trivial3), "NonTrivialDefCtor3 is trivial");
