// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename... Types> struct tuple;

// FIXME: Many more bullets to go

// In a template-argument-list (14.3); the pattern is a template-argument.
template<typename ...Types>
struct tuple_of_refs {
  typedef tuple<Types& ...> types;
};

tuple<int&, float&> *t_int_ref_float_ref;
tuple_of_refs<int&, float&>::types *t_int_ref_float_ref_2 =  t_int_ref_float_ref;

template<typename ...Types>
struct extract_nested_types {
  typedef tuple<typename Types::type...> types;
};

template<typename T>
struct identity {
  typedef T type;
};

tuple<int, float> *t_int_float;
extract_nested_types<identity<int>, identity<float> >::types *t_int_float_2 
  = t_int_float;

// In a dynamic-exception-specification (15.4); the pattern is a type-id.
template<typename ...Types>
struct f_with_except {
  virtual void f() throw(Types...);
};

// FIXME: the code below requires the ability to instantiate pack
// expansions whose pattern is a type-id.
#if 0
struct check_f_with_except_1 : f_with_except<int, float> {
  virtual void f() throw(int, float);
};

struct check_f_with_except_2 : f_with_except<int, float> {
  virtual void f() throw(int);
};
#endif
