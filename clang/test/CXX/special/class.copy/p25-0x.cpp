// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename T, bool B> struct trivially_assignable_check {
  static_assert(B == __has_trivial_assign(T), "");
  static_assert(B == __is_trivially_assignable(T&, T), "");
  static_assert(B == __is_trivially_assignable(T&, const T &), "");
  static_assert(B == __is_trivially_assignable(T&, T &&), "");
  static_assert(B == __is_trivially_assignable(T&&, T), "");
  static_assert(B == __is_trivially_assignable(T&&, const T &), "");
  static_assert(B == __is_trivially_assignable(T&&, T &&), "");
  typedef void type;
};
template<typename T> using trivially_assignable =
  typename trivially_assignable_check<T, true>::type;
template<typename T> using not_trivially_assignable =
  typename trivially_assignable_check<T, false>::type;

struct Trivial {};
using _ = trivially_assignable<Trivial>;

// A copy/move assignment operator for class X is trivial if it is not user-provided,
struct UserProvided {
  UserProvided &operator=(const UserProvided &);
};
using _ = not_trivially_assignable<UserProvided>;

// its declared parameter type is the same as if it had been implicitly
// declared,
struct NonConstCopy {
  NonConstCopy &operator=(NonConstCopy &) = default;
};
using _ = not_trivially_assignable<NonConstCopy>;

// class X has no virtual functions
struct VFn {
  virtual void f();
};
using _ = not_trivially_assignable<VFn>;

// and no virtual base classes
struct VBase : virtual Trivial {};
using _ = not_trivially_assignable<VBase>;

// and the assignment operator selected to copy/move each [direct subobject] is trivial
struct TemplateCtor {
  template<typename T> TemplateCtor operator=(T &);
};
using _ = trivially_assignable<TemplateCtor>;
struct TemplateCtorMember {
  TemplateCtor tc;
};
using _ = trivially_assignable<TemplateCtorMember>;
struct MutableTemplateCtorMember {
  mutable TemplateCtor mtc;
};
// FIXME: This is wrong! The "trivial" copy constructor calls the templated
//        constructor for the mutable member.
static_assert(!__is_trivially_assignable(MutableTemplateCtorMember, const MutableTemplateCtorMember &), ""); // expected-error {{}}
static_assert(__is_trivially_assignable(MutableTemplateCtorMember, MutableTemplateCtorMember &&), "");

// Both trivial and non-trivial special members.
struct TNT {
  TNT &operator=(const TNT &) = default; // trivial
  TNT &operator=(TNT &); // non-trivial

  TNT &operator=(TNT &&) = default; // trivial
  TNT &operator=(const TNT &&); // non-trivial
};

static_assert(!__has_trivial_assign(TNT), "lie deliberately for gcc compatibility");
static_assert(__is_trivially_assignable(TNT, TNT), "");
static_assert(!__is_trivially_assignable(TNT, TNT &), "");
static_assert(__is_trivially_assignable(TNT, const TNT &), "");
static_assert(!__is_trivially_assignable(TNT, volatile TNT &), "");
static_assert(__is_trivially_assignable(TNT, TNT &&), "");
static_assert(!__is_trivially_assignable(TNT, const TNT &&), "");
static_assert(!__is_trivially_assignable(TNT, volatile TNT &&), "");

// This has only trivial special members.
struct DerivedFromTNT : TNT {};

static_assert(__has_trivial_assign(DerivedFromTNT), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT &), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, const DerivedFromTNT &), "");
static_assert(!__is_trivially_assignable(DerivedFromTNT, volatile DerivedFromTNT &), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT &&), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, const DerivedFromTNT &&), "");
static_assert(!__is_trivially_assignable(DerivedFromTNT, volatile DerivedFromTNT &&), "");

// This has only trivial special members.
struct TNTMember {
  TNT tnt;
};

static_assert(__has_trivial_assign(TNTMember), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember &), "");
static_assert(__is_trivially_assignable(TNTMember, const TNTMember &), "");
static_assert(!__is_trivially_assignable(TNTMember, volatile TNTMember &), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember &&), "");
static_assert(__is_trivially_assignable(TNTMember, const TNTMember &&), "");
static_assert(!__is_trivially_assignable(TNTMember, volatile TNTMember &&), "");

struct NCCTNT : NonConstCopy, TNT {};

static_assert(!__has_trivial_assign(NCCTNT), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, const NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, volatile NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT &&), "");
static_assert(!__is_trivially_assignable(NCCTNT, const NCCTNT &&), "");
static_assert(!__is_trivially_assignable(NCCTNT, volatile NCCTNT &&), "");

struct MultipleTrivial {
  // All four of these are trivial.
  MultipleTrivial &operator=(const MultipleTrivial &) & = default;
  MultipleTrivial &operator=(const MultipleTrivial &) && = default;
  MultipleTrivial &operator=(MultipleTrivial &&) & = default;
  MultipleTrivial &operator=(MultipleTrivial &&) && = default;
};

using _ = trivially_assignable<MultipleTrivial>;

struct RefQualifier {
  RefQualifier &operator=(const RefQualifier &) & = default;
  RefQualifier &operator=(const RefQualifier &) &&;
  RefQualifier &operator=(RefQualifier &&) &;
  RefQualifier &operator=(RefQualifier &&) && = default;
};
struct DerivedFromRefQualifier : RefQualifier {
  // Both of these call the trivial copy operation.
  DerivedFromRefQualifier &operator=(const DerivedFromRefQualifier &) & = default;
  DerivedFromRefQualifier &operator=(const DerivedFromRefQualifier &) && = default;
  // Both of these call the non-trivial move operation.
  DerivedFromRefQualifier &operator=(DerivedFromRefQualifier &&) & = default;
  DerivedFromRefQualifier &operator=(DerivedFromRefQualifier &&) && = default;
};
static_assert(__is_trivially_assignable(DerivedFromRefQualifier&, const DerivedFromRefQualifier&), "");
static_assert(__is_trivially_assignable(DerivedFromRefQualifier&&, const DerivedFromRefQualifier&), "");
static_assert(!__is_trivially_assignable(DerivedFromRefQualifier&, DerivedFromRefQualifier&&), "");
static_assert(!__is_trivially_assignable(DerivedFromRefQualifier&&, DerivedFromRefQualifier&&), "");
