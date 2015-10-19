// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style=LLVM -lines=1:1 -lines=5:5 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^int\ \*i;$}}
  int*i;

// CHECK: {{^\ \ int\ \ \*\ \ i;$}}
  int  *  i; 

// CHECK: {{^\ \ int\ \*i;$}}
  int   *   i;
