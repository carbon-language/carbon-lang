// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

enum Color {
  Red,
  Orange,
  Yellow,
  Green,
  Blue,
  Indigo,
  Violet
};

void test(enum Color color) {
  switch (color) {
    case Red:
      break;
      
    case Yellow:
      break;

    // CHECK-CC1: Blue : 0
    // CHECK-NEXT-CC1: Green : 0
    // CHECK-NEXT-CC1: Indigo : 0
    // CHECK-NEXT-CC1: Orange : 0
    // CHECK-NEXT-CC1: Violet : 0
    case 
