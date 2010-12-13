// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s


// An appearance of a name of a parameter pack that is not expanded is
// ill-formed.
template<typename ... Types>
struct TestPPName 
  : public Types  // expected-error{{base type contains unexpanded parameter pack}}
{
  typedef Types *types_pointer; // expected-error{{declaration type contains unexpanded parameter pack}}
};
