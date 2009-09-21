// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

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
      // CHECK-NEXT-CC1: Blue : 0 : N::C::Blue
      // CHECK-NEXT-CC1: Green : 0 : N::C::Green
      // CHECK-NEXT-CC1: Indigo : 0 : N::C::Indigo
      // CHECK-NEXT-CC1: Orange : 0 : N::C::Orange
      // CHECK-NEXT-CC1: Red : 0 : N::C::Red
      // CHECK-NEXT-CC1: Violet : 0 : N::C::Violet
      // CHECK-NEXT-CC1: Yellow : 0 : N::C::Yellow
    case 
