// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-format -i %t.cpp
// RUN: FileCheck -input-file=%t.cpp %s

// CHECK: {{^int \*i;}}
 int   *  i  ;
