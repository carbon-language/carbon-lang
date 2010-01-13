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
      
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:19:10 %s -o - | FileCheck -check-prefix=CC1 %s
    // CHECK-CC1: Blue
    // CHECK-CC1-NEXT: Green
    // CHECK-CC1-NEXT: Indigo
    // CHECK-CC1-NEXT: Orange
    // CHECK-CC1-NEXT: Violet
      
