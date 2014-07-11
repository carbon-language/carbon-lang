// Check handling -mhard-float / -msoft-float options
// when build for SPARC platforms.
//
// Default sparc
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// CHECK-DEF: "-msoft-float"
//
// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD: "-mhard-float"
//
// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: "-msoft-float"
//
// Default sparc64
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF-SPARC64 %s
// CHECK-DEF-SPARC64: "-msoft-float"
//
// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD-SPARC64 %s
// CHECK-HARD-SPARC64: "-mhard-float"
//
// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-SPARC64 %s
// CHECK-SOFT-SPARC64: "-msoft-float"
