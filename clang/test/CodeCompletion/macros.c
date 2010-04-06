enum Color {
  Red, Green, Blue
};

struct Point {
  float x, y, z;
  enum Color color;
};

void test(struct Point *p) {
  // RUN: %clang_cc1 -include %S/Inputs/macros.h -fsyntax-only -code-completion-macros -code-completion-at=%s:12:14 %s -o - | FileCheck -check-prefix=CC1 %s
  switch (p->IDENTITY(color)) {
  // RUN: %clang_cc1 -include %S/Inputs/macros.h -fsyntax-only -code-completion-macros -code-completion-at=%s:14:9 %s -o - | FileCheck -check-prefix=CC2 %s
    case 
  }

  // Run the same tests, this time with macros loaded from the PCH file.
  // RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/macros.h
  // RUN: %clang_cc1 -include-pch %t -fsyntax-only -code-completion-macros -code-completion-at=%s:12:14 %s -o - | FileCheck -check-prefix=CC1 %s
  // RUN: %clang_cc1 -include-pch %t -fsyntax-only -code-completion-macros -code-completion-at=%s:14:9 %s -o - | FileCheck -check-prefix=CC2 %s

  // CC1: color
  // CC1: x
  // CC1: y
  // CC1: z

  // CC2: BAR(<#X#>, <#Y#>)
  // CC2: Blue
  // CC2: FOO
  // CC2: Green
  // CC2: IDENTITY(<#X#>)
  // CC2: Red
  // CC2: WIBBLE
}
