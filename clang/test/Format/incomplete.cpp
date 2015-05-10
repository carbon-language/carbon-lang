// RUN: grep -Ev "// *[A-Z-]+:" %s > %t2.cpp
// RUN: clang-format -style=LLVM %t2.cpp -cursor=0 > %t.cpp
// RUN: FileCheck -strict-whitespace -input-file=%t.cpp %s
// CHECK: {{"IncompleteFormat": true}}
// CHECK: {{^int\ \i;$}}
 int    i;
// CHECK: {{^f  \( g  \(;$}}
f  ( g  (;
