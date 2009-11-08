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
    
  case 
    // RUN: clang-cc -fsyntax-only -code-completion-at=%s:21:8 %s -o - | FileCheck -check-prefix=CC1 %s
    // CHECK-CC1: Blue : 0 : N::Blue
    // CHECK-CC1-NEXT: Green : 0 : N::Green
    // CHECK-CC1-NEXT: Indigo : 0 : N::Indigo
    // CHECK-CC1-NEXT: Orange : 0 : N::Orange
    // CHECK-CC1-NEXT: Violet : 0 : N::Violet
    
    // RUN: true
