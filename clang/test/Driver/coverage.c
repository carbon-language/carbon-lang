// Test coverage flag.
//
// RUN: %clang_cl -### -coverage %s -o foo/bar.o 2>&1 | FileCheck -check-prefix=CLANG-CL-COVERAGE %s
// CLANG-CL-COVERAGE-NOT: error:
// CLANG-CL-COVERAGE-NOT: warning:
// CLANG-CL-COVERAGE-NOT: argument unused
// CLANG-CL-COVERAGE-NOT: unknown argument
