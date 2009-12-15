// RUN: %clang_cc1 -fsyntax-only -verify %s

class A {};

namespace B {
  namespace A {}
  using namespace A ;
}

namespace C {}

namespace D {
  
  class C {
    
    using namespace B ; // expected-error{{not allowed}}
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
using namespace ::A // expected-error{{expected namespace name}} \
                    // expected-error{{expected ';' after namespace name}}
                    B ; 

void test_nslookup() {
  int B;
  class C;
  using namespace B;
  using namespace C;
}

