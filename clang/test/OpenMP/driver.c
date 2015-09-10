// Test that by default -fnoopenmp-use-tls is passed to frontend.
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT: -cc1
// CHECK-DEFAULT-NOT: -fnoopenmp-use-tls
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp -fnoopenmp-use-tls | FileCheck --check-prefix=CHECK-NO-TLS %s
// CHECK-NO-TLS: -cc1
// CHECK-NO-TLS-SAME: -fnoopenmp-use-tls
//
