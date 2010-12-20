// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename... Types> struct Tuple;

// FIXME: Many more bullets to go

// In a template-argument-list (14.3); the pattern is a template-argument.
template<typename ...Types>
struct tuple_of_refs {
  typedef Tuple<Types& ...> types;
};

Tuple<int&, float&> *t_int_ref_float_ref;
tuple_of_refs<int&, float&>::types *t_int_ref_float_ref_2 =  t_int_ref_float_ref;
  
