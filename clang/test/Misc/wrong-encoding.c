// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value %s 2>&1 | FileCheck -strict-whitespace %s

void foo() {

  "§Ã"; // ø
// CHECK: {{^  "<A7><C3>"; // <F8>}}
// CHECK: {{^   \^~~~~~~}}

  /* þ« */ const char *d = "¥";

// CHECK: {{^  /\* <FE><AB> \*/ const char \*d = "<A5>";}}
// CHECK: {{^                                  \^~~~}}

  "xxé¿¿¿d";
// CHECK: {{^  "xx<U\+9FFF><BF>d";}}
// CHECK: {{^             \^~~~}}

  "xxé¿bcd";
// CHECK: {{^  "xx<E9><BF>bcd";}}
// CHECK: {{^     \^~~~~~~~}}

  "xxéabcd";
// CHECK: {{^  "xx<E9>abcd";}}
// CHECK: {{^     \^~~~}}

  "xxé¿é¿d";
// CHECK: {{^  "xx<E9><BF><E9><BF>d";}}
// CHECK: {{^     \^~~~~~~~~~~~~~~}}

  "xxé¿xxxxxxxxxxxxxxxxxxxxxé¿xx";
// CHECK: {{^  "xx<E9><BF>xxxxxxxxxxxxxxxxxxxxx<E9><BF>xx";}}
// CHECK: {{^     \^~~~~~~~                     ~~~~~~~~}}
}
