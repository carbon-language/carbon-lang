// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-format -style=LLVM -offset=2 -length=0 -offset=28 -length=0 -i %t.cpp
// RUN: FileCheck -strict-whitespace -input-file=%t.cpp %s
// CHECK: {{^int\ \*i;$}}
  int*i;

// CHECK: {{^\ \ int\ \ \*\ \ i;$}}
  int  *  i; 

// CHECK: {{^\ \ int\ \*i;$}}
  int   *   i;
