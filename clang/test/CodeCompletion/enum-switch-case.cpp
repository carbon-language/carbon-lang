// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace N {
  enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Indigo,
    Violet
  };
}

void test(enum N::Color color) {
  switch (color) {
    case N::Red:
      break;
      
    case N::Yellow:
      break;
      
      // CHECK-CC1: Blue : 0 : N::Blue
      // CHECK-NEXT-CC1: Green : 0 : N::Green
      // CHECK-NEXT-CC1: Indigo : 0 : N::Indigo
      // CHECK-NEXT-CC1: Orange : 0 : N::Orange
      // CHECK-NEXT-CC1: Violet : 0 : N::Violet
    case 
