// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s

void foo() {

  "§Ã"; // ø
// CHECK: {{^  "<A7><C3>"; // <F8>}}
// CHECK: {{^  \^}}

  /* þ« */ const char *d = "¥";

// CHECK: {{^  /\* <FE><AB> \*/ const char \*d = "<A5>";}}
// CHECK: {{^                                 \^}}

// CHECK: {{^  "<A7><C3>"; // <F8>}}
// CHECK: {{^  \^~~~~~~~~~}}
}
