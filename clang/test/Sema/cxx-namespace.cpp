// RUN: clang -fsyntax-only -verify %s 
namespace A { // expected-error {{error: previous definition is here}}
  int A;
  void f() { A = 0; }
}

void f() { A = 0; } // expected-error {{error: unexpected namespace name 'A': expected expression}}
int A; // expected-error {{error: redefinition of 'A' as different kind of symbol}}
class A; // expected-error {{error: redefinition of 'A' as different kind of symbol}}

class B {}; // expected-error {{error: previous definition is here}}
namespace B {} // expected-error {{error: redefinition of 'B' as different kind of symbol}}

void C(); // expected-error {{error: previous definition is here}}
namespace C {} // expected-error {{error: redefinition of 'C' as different kind of symbol}}

namespace D {
  class D {};
}

namespace S1 {
  int x;

  namespace S2 {

    namespace S3 {
      B x;
    }
  }
}

namespace S1 {
  void f() {
    x = 0;
  }

  namespace S2 {
    
    namespace S3 {
      void f() {
        x = 0; // expected-error {{error: incompatible type assigning 'int', expected 'class B'}}
      }
    }

    int y;
  }
}

namespace S1 {
  namespace S2 {
    namespace S3 {
      void f3() {
        y = 0;
      }
    }
  }
}
