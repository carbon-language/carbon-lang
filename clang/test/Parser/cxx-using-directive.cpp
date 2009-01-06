// RUN: clang -fsyntax-only -verify %s

class A {};

namespace B {
  namespace A {}
  using namespace A ;
}

namespace C {}

namespace D {
  
  class C {
    
    using namespace B ; // expected-error{{expected unqualified-id}}
    //FIXME: this needs better error message
  };
  
  namespace B {}
  
  using namespace C ;
  using namespace B::A ; // expected-error{{expected namespace name}}
  //FIXME: would be nice to note, that A is not member of D::B
  using namespace ::B::A ;
  using namespace ::D::C ; // expected-error{{expected namespace name}}
}

using namespace ! ; // expected-error{{expected namespace name}}
using namespace A ; // expected-error{{expected namespace name}}
using namespace ::A // expected-error{{expected namespace name}}
                    B ; // expected-error{{expected ';' after namespace name}}

