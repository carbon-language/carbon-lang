// RUN: grep -Ev "// *[A-Z-]+:" %s > %t2.cpp
// RUN: clang-format -style=LLVM %t2.cpp -cursor=6 > %t.cpp
// RUN: FileCheck -strict-whitespace -input-file=%t.cpp %s
// CHECK: {{^\{ "Cursor": 4 \}$}}
// CHECK: {{^int\ \i;$}}
 int    i;
