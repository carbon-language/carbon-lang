@import cxx_templates_common;

template<typename T> T f();
template<typename T> T f(T t) { return t; }
namespace N {
  template<typename T> T f();
  template<typename T> T f(T t) { return t; }
}

template<typename> int template_param_kinds_1();
template<template<typename, int, int...> class> int template_param_kinds_2();
template<template<typename T, typename U, U> class> int template_param_kinds_3();

template<typename T> struct SomeTemplate<T&> {};
template<typename T> struct SomeTemplate<T&>;
typedef SomeTemplate<int&> SomeTemplateIntRef;

extern DefinedInCommon &defined_in_common;

@import cxx_templates_b_impl;

template<typename T, typename> struct Identity { typedef T type; };
template<typename T> void UseDefinedInBImpl() {
  typename Identity<DefinedInBImpl, T>::type dependent;
  FoundByADL(dependent);
  typename Identity<DefinedInBImpl, T>::type::Inner inner;
  dependent.f();
}

extern DefinedInBImpl &defined_in_b_impl;

@import cxx_templates_a;
template<typename T> void UseDefinedInBImplIndirectly(T &v) {
  PerformDelayedLookup(v);
}

void TriggerInstantiation() {
  UseDefinedInBImpl<void>();
}
