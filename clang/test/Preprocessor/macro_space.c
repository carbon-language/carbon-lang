// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#define XX
! XX,

// CHECK: {{^}}! ,{{$}}
