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
