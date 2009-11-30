// RUN: clang-cc -fsyntax-only -ftemplate-depth 5 -verify %s

template<typename T> struct X : X<T*> { }; // expected-error{{recursive template instantiation exceeded maximum depth of 5}} \
// expected-note{{use -ftemplate-depth-N to increase recursive template instantiation depth}} \
// expected-note 5 {{instantiation of template class}}

void test() { 
  (void)sizeof(X<int>); // expected-note {{instantiation of template class}}
}
