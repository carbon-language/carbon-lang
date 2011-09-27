// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++0x-extensions %s

namespace fizbin { class Foobar {}; } // expected-note 2 {{'fizbin::Foobar' declared here}} \
                                      // expected-note {{'Foobar' declared here}}
Foobar *my_bar  // expected-error{{unknown type name 'Foobar'; did you mean 'fizbin::Foobar'?}}
    = new Foobar; // expected-error{{unknown type name 'Foobar'; did you mean 'fizbin::Foobar'?}}
fizbin::Foobar *my_foo = new fizbin::FooBar; // expected-error{{unknown type name 'FooBar'; did you mean 'Foobar'?}}

namespace barstool { int toFoobar() { return 1; } } // expected-note 3 {{'barstool::toFoobar' declared here}}
int Double(int x) { return x + x; }
void empty() {
  Double(toFoobar()); // expected-error{{{use of undeclared identifier 'toFoobar'; did you mean 'barstool::toFoobar'?}}
}

namespace fizbin {
  namespace baztool { bool toFoobar() { return true; } } // expected-note{{'fizbin::baztool' declared here}}
  namespace nested { bool moreFoobar() { return true; } } // expected-note{{'fizbin::nested::moreFoobar' declared here}}
  namespace nested { bool lessFoobar() { return true; } } // expected-note{{'fizbin::nested' declared here}} \
                                                          // expected-note{{'fizbin::nested::lessFoobar' declared here}}
  class dummy { // expected-note 2 {{'fizbin::dummy' declared here}}
   public:
    static bool moreFoobar() { return false; } // expected-note{{'moreFoobar' declared here}}
  };
}
void Check() { // expected-note{{'Check' declared here}}
  if (toFoobar()) Double(7); // expected-error{{use of undeclared identifier 'toFoobar'; did you mean 'barstool::toFoobar'?}}
  if (noFoobar()) Double(7); // expected-error{{use of undeclared identifier 'noFoobar'; did you mean 'barstool::toFoobar'?}}
  if (moreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'moreFoobar'; did you mean 'fizbin::nested::moreFoobar'}}
  if (lessFoobar()) Double(7); // expected-error{{use of undeclared identifier 'lessFoobar'; did you mean 'fizbin::nested::lessFoobar'?}}
  if (baztool::toFoobar()) Double(7); // expected-error{{use of undeclared identifier 'baztool'; did you mean 'fizbin::baztool'?}}
  if (nested::moreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'nested'; did you mean 'fizbin::nested'?}}
  if (dummy::moreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'dummy'; did you mean 'fizbin::dummy'?}}
  if (dummy::mreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'dummy'; did you mean 'fizbin::dummy'?}} \
                                     // expected-error{{no member named 'mreFoobar' in 'fizbin::dummy'; did you mean 'moreFoobar'?}}
  if (moFoobin()) Double(7); // expected-error{{use of undeclared identifier 'moFoobin'}}
}

void Alt() {
  Cleck(); // expected-error{{{use of undeclared identifier 'Cleck'; did you mean 'Check'?}}
}

namespace N {
  namespace inner {
    class myvector { /* ... */ }; // expected-note{{'inner::myvector' declared here}}
  }

  void f() {
    myvector v; // expected-error{{unknown type name 'myvector'; did you mean 'inner::myvector'?}}
  }
}

namespace realstd {
  inline namespace __1 {
    class mylinkedlist { /* ... */ }; // expected-note 2 {{'realstd::mylinkedlist' declared here}}
  }

  class linkedlist { /* ... */ };
}

void f() {
  mylinkedlist v; // expected-error{{unknown type name 'mylinkedlist'; did you mean 'realstd::mylinkedlist'?}}
  nylinkedlist w; // expected-error{{unknown type name 'nylinkedlist'; did you mean 'realstd::mylinkedlist'?}}
}

// Test case from http://llvm.org/bugs/show_bug.cgi?id=10318
namespace llvm {
 template <typename T> class GraphWriter {}; // expected-note {{'llvm::GraphWriter' declared here}} \
                                             // expected-note {{'GraphWriter' declared here}}
}

struct S {};
void bar() {
 GraphWriter<S> x; //expected-error{{no template named 'GraphWriter'; did you mean 'llvm::GraphWriter'?}}
 (void)new llvm::GraphWriter; // expected-error {{expected a type}}
 (void)new llvm::Graphwriter<S>; // expected-error {{no template named 'Graphwriter' in namespace 'llvm'; did you mean 'GraphWriter'?}}
}
