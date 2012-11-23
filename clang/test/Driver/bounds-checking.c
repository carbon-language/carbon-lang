// RUN: %clang -fsanitize=bounds -### -fsyntax-only %s 2> %t
// RUN: FileCheck -check-prefix=CHECK < %t %s
// CHECK: "-fsanitize=bounds"

// RUN: %clang -fbounds-checking -### -fsyntax-only %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OLD < %t %s
// CHECK-OLD: "-fsanitize=bounds"

// RUN: %clang -fbounds-checking=3 -### -fsyntax-only %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OLD2 < %t %s
// CHECK-OLD2: "-fsanitize=bounds"
