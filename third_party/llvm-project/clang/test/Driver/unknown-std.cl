// This file checks output given when processing OpenCL files.
// When user selects invalid language standard
// print out supported values with short description.

// RUN: not %clang %s -std=foobar -c 2>&1 | \
// RUN: FileCheck --match-full-lines %s

// CHECK: error: invalid value 'foobar' in '-std=foobar'
// CHECK-NEXT: note: use 'cl1.0' for 'OpenCL 1.0' standard
// CHECK-NEXT: note: use 'cl1.1' for 'OpenCL 1.1' standard
// CHECK-NEXT: note: use 'cl1.2' for 'OpenCL 1.2' standard
// CHECK-NEXT: note: use 'cl2.0' for 'OpenCL 2.0' standard
// CHECK-NEXT: note: use 'cl3.0' for 'OpenCL 3.0' standard
// CHECK-NEXT: note: use 'clc++1.0' or 'clc++' for 'C++ for OpenCL 1.0' standard
// CHECK-NEXT: note: use 'clc++2021' for 'C++ for OpenCL 2021' standard

// Make sure that no other output is present.
// CHECK-NOT: {{^.+$}}

