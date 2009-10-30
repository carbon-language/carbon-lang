#define FOO
#define BAR(X, Y) X, Y
#define IDENTITY(X) X
#define WIBBLE(...)

enum Color {
  Red, Green, Blue
};

struct Point {
  float x, y, z;
  enum Color color;
};

void test(struct Point *p) {
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:17:14 %s -o - | FileCheck -check-prefix=CC1 %s &&
  switch (p->IDENTITY(color)) {
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:9 %s -o - | FileCheck -check-prefix=CC2 %s &&
    case 
  }
  // CC1: color
  // CC1: x
  // CC1: y
  // CC1: z
  // CC1: BAR(<#X#>, <#Y#>)
  // CC1: FOO
  // CC1: IDENTITY(<#X#>)
  // CC1: WIBBLE
  // CC2: Blue
  // CC2: Green
  // CC2: Red
  // CC2: BAR(<#X#>, <#Y#>)
  // CC2: FOO
  // CC2: IDENTITY(<#X#>)
  // CC2: WIBBLE
  // RUN: true
}
