// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s


// An appearance of a name of a parameter pack that is not expanded is
// ill-formed.
template<typename ... Types>
struct TestPPName 
  : public Types  // expected-error{{base type contains unexpanded parameter pack 'Types'}}
{
  typedef Types *types_pointer; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
};

template<typename ... Types>
void TestPPNameFunc(int i) {
  f(static_cast<Types>(i)); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
}

template<typename T, typename U> struct pair;

template<typename ...OuterTypes>
struct MemberTemplatePPNames {
  template<typename ...InnerTypes>
  struct Inner {
    typedef pair<OuterTypes, InnerTypes>* types; // expected-error{{declaration type contains unexpanded parameter packs 'OuterTypes' and 'InnerTypes'}}

    template<typename ...VeryInnerTypes>
    struct VeryInner {
      typedef pair<pair<VeryInnerTypes, OuterTypes>, pair<InnerTypes, OuterTypes> > types; // expected-error{{declaration type contains unexpanded parameter packs 'VeryInnerTypes', 'OuterTypes', ...}}
    };
  };
};
