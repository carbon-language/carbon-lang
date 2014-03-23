@import cxx_templates_common;

template<typename T> T f() { return T(); }
template<typename T> T f(T);
namespace N {
  template<typename T> T f() { return T(); }
  template<typename T> T f(T);
}

template<int N> int template_param_kinds_1();
template<template<typename T, int, int> class> int template_param_kinds_2();
template<template<typename T, typename U, T> class> int template_param_kinds_3();

template<typename T> struct SomeTemplate<T*>;
template<typename T> struct SomeTemplate<T*> {};
typedef SomeTemplate<int*> SomeTemplateIntPtr;

template<typename T> void PerformDelayedLookup(T &t) {
  t.f();
  typename T::Inner inner;
  FoundByADL(t);
}

template<typename T> void PerformDelayedLookupInDefaultArgument(T &t, int a = (FoundByADL(T()), 0)) {}

template<typename T> struct RedeclaredAsFriend {};

void use_some_template_a() {
  SomeTemplate<char[2]> a;
  SomeTemplate<char[1]> b, c;
  b = c;
}

template<int> struct MergeTemplates;
MergeTemplates<0> *merge_templates_a;

auto enum_a_from_a = CommonTemplate<int>::a;
const auto enum_c_from_a = CommonTemplate<int>::c;

template<int> struct UseInt;
template<typename T> void UseRedeclaredEnum(UseInt<T() + CommonTemplate<char>::a>);
constexpr void (*UseRedeclaredEnumA)(UseInt<1>) = UseRedeclaredEnum<int>;

template<typename> struct MergeSpecializations;
template<typename T> struct MergeSpecializations<T*> {
  typedef int partially_specialized_in_a;
};
template<> struct MergeSpecializations<char> {
  typedef int explicitly_specialized_in_a;
};

void InstantiateWithFriend(Std::WithFriend<int> wfi) {}
