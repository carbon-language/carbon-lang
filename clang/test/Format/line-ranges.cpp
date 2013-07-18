// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-format -style=LLVM -lines=1:1 -lines=5:5 -i %t.cpp
// RUN: FileCheck -strict-whitespace -input-file=%t.cpp %s
// CHECK: {{^int\ \*i;$}}
  int*i;

// CHECK: {{^\ \ int\ \ \*\ \ i;$}}
  int  *  i; 

// CHECK: {{^\ \ int\ \*i;$}}
  int   *   i;
