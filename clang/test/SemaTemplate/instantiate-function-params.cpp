// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR6619
template<bool C> struct if_c { };
template<typename T1> struct if_ {
  typedef if_c< static_cast<bool>(T1::value)> almost_type_; // expected-note 7{{in instantiation}}
};
template <class Model, void (Model::*)()> struct wrap_constraints { };
template <class Model> 
inline char has_constraints_(Model* ,  // expected-note 4{{while substituting}} \
                             // expected-note 3{{candidate template ignored}}
                               wrap_constraints<Model,&Model::constraints>* = 0); // expected-note 4{{in instantiation}}

template <class Model> struct not_satisfied {
  static const bool value = sizeof( has_constraints_((Model*)0)  == 1); // expected-error 3{{no matching function}}
};
template <class ModelFn> struct requirement_;
template <void(*)()> struct instantiate {
};
template <class Model> struct requirement_<void(*)(Model)>                           : if_<       not_satisfied<Model>         >::type { // expected-error 3{{no type named}} \
  // expected-note 7{{in instantiation}}
};
template <class Model> struct usage_requirements {
};
template < typename TT > struct InputIterator                            {
    typedef  instantiate< & requirement_<void(*)(usage_requirements<InputIterator> x)>::failed> boost_concept_check1; // expected-note 2{{in instantiation}}
};
template < typename TT > struct ForwardIterator                              : InputIterator<TT>                              { // expected-note 2{{in instantiation}}
  typedef instantiate< & requirement_<void(*)(usage_requirements<ForwardIterator> x)>::failed> boost_concept_check2; // expected-note 2 {{in instantiation}}

};
typedef instantiate< &requirement_<void(*)(ForwardIterator<char*> x)>::failed> boost_concept_checkX; // expected-error{{no member named}} \
// expected-note 6{{in instantiation}}
