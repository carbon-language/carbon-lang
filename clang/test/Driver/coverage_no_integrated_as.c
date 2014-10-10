// REQUIRES: clang-driver
// XFAIL: win32,win64

// RUN: %clang -### -S -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s 2>&1 | FileCheck -check-prefix=CHECK-GCNO-DEFAULT-LOCATION %s

// RUN: %clang -### -S -fprofile-arcs %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s -o /foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION %s

// RUN: %clang -### -S -fprofile-arcs %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -S -fprofile-arcs -no-integrated-as %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -c -fprofile-arcs %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s
// RUN: %clang -### -c -fprofile-arcs -no-integrated-as %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CHECK-GCNO-LOCATION-REL-PATH %s


// CHECK-GCNO-DEFAULT-LOCATION: "-coverage-file" "{{.*}}/coverage_no_integrated_as.c"
// CHECK-GCNO-DEFAULT-LOCATION-NOT: "-coverage-file" "/tmp/{{.*}}/coverage_no_integrated_as.c"
// CHECK-GCNO-LOCATION: "-coverage-file" "/foo/bar.o"
// CHECK-GCNO-LOCATION-REL-PATH: "-coverage-file" "{{.*}}/foo/bar.o"
