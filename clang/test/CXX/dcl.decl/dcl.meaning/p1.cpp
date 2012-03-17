// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR8019 {
  struct x;
  template<typename T> struct x2;
  struct y { 
    struct PR8019::x { int x; };  // expected-error{{non-friend class member 'x' cannot have a qualified name}}
  
    struct inner;
    struct y::inner { }; // expected-warning{{extra qualification on member 'inner'}}

    template<typename T>
    struct PR8019::x2 { }; // expected-error{{non-friend class member 'x2' cannot have a qualified name}}

    template<typename T>
    struct inner_template;
  
    template<typename T>
    struct y::inner_template { }; // expected-warning{{extra qualification on member 'inner_template'}}
  };

}
