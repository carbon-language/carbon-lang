// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename T, bool B> struct trivially_copyable_check {
  static_assert(B == __has_trivial_copy(T), "");
  static_assert(B == __is_trivially_constructible(T, T), "");
  static_assert(B == __is_trivially_constructible(T, const T &), "");
  static_assert(B == __is_trivially_constructible(T, T &&), "");
  typedef void type;
};
template<typename T> using trivially_copyable =
  typename trivially_copyable_check<T, true>::type;
template<typename T> using not_trivially_copyable =
  typename trivially_copyable_check<T, false>::type;

struct Trivial {};
using _ = trivially_copyable<Trivial>;

// A copy/move constructor for class X is trivial if it is not user-provided,
struct UserProvided {
  UserProvided(const UserProvided &);
};
using _ = not_trivially_copyable<UserProvided>;

// its declared parameter type is the same as if it had been implicitly
// declared,
struct NonConstCopy {
  NonConstCopy(NonConstCopy &) = default;
};
using _ = not_trivially_copyable<NonConstCopy>;

// class X has no virtual functions
struct VFn {
  virtual void f();
};
using _ = not_trivially_copyable<VFn>;

// and no virtual base classes
struct VBase : virtual Trivial {};
using _ = not_trivially_copyable<VBase>;

// and the constructor selected to copy/move each [direct subobject] is trivial
struct TemplateCtor {
  template<typename T> TemplateCtor(T &);
};
using _ = trivially_copyable<TemplateCtor>;
struct TemplateCtorMember {
  TemplateCtor tc;
};
using _ = trivially_copyable<TemplateCtorMember>;

// We can select a non-trivial copy ctor even if there is a trivial one.
struct MutableTemplateCtorMember {
  mutable TemplateCtor mtc;
};
// FIXME: This is wrong! The "trivial" copy constructor calls the templated
//        constructor for the mutable member.
static_assert(!__is_trivially_constructible(MutableTemplateCtorMember, const MutableTemplateCtorMember &), ""); // expected-error {{}}
static_assert(__is_trivially_constructible(MutableTemplateCtorMember, MutableTemplateCtorMember &&), "");
struct MutableTemplateCtorMember2 {
  MutableTemplateCtorMember2(const MutableTemplateCtorMember2 &) = default;
  MutableTemplateCtorMember2(MutableTemplateCtorMember2 &&) = default;
  mutable TemplateCtor mtc;
};
static_assert(!__is_trivially_constructible(MutableTemplateCtorMember2, const MutableTemplateCtorMember2 &), "");
static_assert(__is_trivially_constructible(MutableTemplateCtorMember2, MutableTemplateCtorMember2 &&), "");

// Both trivial and non-trivial special members.
struct TNT {
  TNT(const TNT &) = default; // trivial
  TNT(TNT &); // non-trivial

  TNT(TNT &&) = default; // trivial
  TNT(const TNT &&); // non-trivial
};

static_assert(!__has_trivial_copy(TNT), "lie deliberately for gcc compatibility");
static_assert(__is_trivially_constructible(TNT, TNT), "");
static_assert(!__is_trivially_constructible(TNT, TNT &), "");
static_assert(__is_trivially_constructible(TNT, const TNT &), "");
static_assert(!__is_trivially_constructible(TNT, volatile TNT &), "");
static_assert(__is_trivially_constructible(TNT, TNT &&), "");
static_assert(!__is_trivially_constructible(TNT, const TNT &&), "");
static_assert(!__is_trivially_constructible(TNT, volatile TNT &&), "");

// This has only trivial special members.
struct DerivedFromTNT : TNT {};

static_assert(__has_trivial_copy(DerivedFromTNT), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT &), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, const DerivedFromTNT &), "");
static_assert(!__is_trivially_constructible(DerivedFromTNT, volatile DerivedFromTNT &), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT &&), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, const DerivedFromTNT &&), "");
static_assert(!__is_trivially_constructible(DerivedFromTNT, volatile DerivedFromTNT &&), "");

// This has only trivial special members.
struct TNTMember {
  TNT tnt;
};

static_assert(__has_trivial_copy(TNTMember), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember &), "");
static_assert(__is_trivially_constructible(TNTMember, const TNTMember &), "");
static_assert(!__is_trivially_constructible(TNTMember, volatile TNTMember &), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember &&), "");
static_assert(__is_trivially_constructible(TNTMember, const TNTMember &&), "");
static_assert(!__is_trivially_constructible(TNTMember, volatile TNTMember &&), "");

struct NCCTNT : NonConstCopy, TNT {};

static_assert(!__has_trivial_copy(NCCTNT), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, const NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, volatile NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT &&), "");
static_assert(!__is_trivially_constructible(NCCTNT, const NCCTNT &&), "");
static_assert(!__is_trivially_constructible(NCCTNT, volatile NCCTNT &&), "");
