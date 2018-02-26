// Check that -dwarf-column-info does not get added to the cc1 line:
// 1) When -gcodeview is present via the clang or clang++ driver
// 2) When /Z7 is present via the cl driver.

// RUN: %clang -### -c -g -gcodeview %s 2> %t1
// RUN: FileCheck < %t1 %s
// RUN: %clangxx -### -c -g -gcodeview %s 2> %t2
// RUN: FileCheck < %t2 %s
// RUN: %clang_cl -### /c /Z7 %s 2> %t2
// RUN: FileCheck < %t2 %s

// CHECK: "-cc1"
// CHECK-NOT: "-dwarf-column-info"
