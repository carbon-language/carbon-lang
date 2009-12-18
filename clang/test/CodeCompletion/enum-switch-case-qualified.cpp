namespace M {
  
namespace N {
  struct C {
    enum Color {
      Red,
      Orange,
      Yellow,
      Green,
      Blue,
      Indigo,
      Violet
    };
  };
}
  
}

namespace M {
  
void test(enum N::C::Color color) {
  switch (color) {
  case 
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:23:8 %s -o - | FileCheck -check-prefix=CC1 %s
    // CHECK-CC1: Blue : 0 : [#enum M::N::C::Color#]N::C::Blue
    // CHECK-CC1-NEXT: Green : 0 : [#enum M::N::C::Color#]N::C::Green
    // CHECK-CC1-NEXT: Indigo : 0 : [#enum M::N::C::Color#]N::C::Indigo
    // CHECK-CC1-NEXT: Orange : 0 : [#enum M::N::C::Color#]N::C::Orange
    // CHECK-CC1-NEXT: Red : 0 : [#enum M::N::C::Color#]N::C::Red
    // CHECK-CC1-NEXT: Violet : 0 : [#enum M::N::C::Color#]N::C::Violet
    // CHECK-CC1: Yellow : 0 : [#enum M::N::C::Color#]N::C::Yellow
      
