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
    // RUN: clang-cc -fsyntax-only -code-completion-at=%s:23:8 %s -o - | FileCheck -check-prefix=CC1 %s &&
    // RUN: true
    // CHECK-CC1: Blue : 0 : N::C::Blue
    // CHECK-CC1-NEXT: Green : 0 : N::C::Green
    // CHECK-CC1-NEXT: Indigo : 0 : N::C::Indigo
    // CHECK-CC1-NEXT: Orange : 0 : N::C::Orange
    // CHECK-CC1-NEXT: Red : 0 : N::C::Red
    // CHECK-CC1-NEXT: Violet : 0 : N::C::Violet
    // CHECK-CC1: Yellow : 0 : N::C::Yellow
      
