// RUN: grep -Ev "// *[A-Z0-9_]+:" %s > %t.js
// RUN: grep -Ev "// *[A-Z0-9_]+:" %s > %t.cpp
// RUN: clang-format -style=llvm %t.js | FileCheck -strict-whitespace -check-prefix=CHECK1 %s
// RUN: clang-format -style=llvm %t.cpp | FileCheck -strict-whitespace -check-prefix=CHECK2 %s
// CHECK1: {{^a >>>= b;$}}
// CHECK2: {{^a >> >= b;$}}
a >>>= b;
