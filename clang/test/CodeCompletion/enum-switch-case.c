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

    case Green:
      break;
      
    // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:10 %s -o - | FileCheck -check-prefix=CC1 %s &&
    // CHECK-CC1: Blue : 0
    // CHECK-CC1-NEXT: Green : 0
    // CHECK-CC1-NEXT: Indigo : 0
    // CHECK-CC1-NEXT: Orange : 0
    // CHECK-CC1-NEXT: Violet : 0
    // RUN: true
      
