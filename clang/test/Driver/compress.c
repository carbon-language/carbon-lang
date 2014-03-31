// RUN: %clang -c %s -Wa,--compress-debug-sections 2>&1 | FileCheck %s
// REQUIRES: nozlib

// CHECK: warning: cannot compress debug sections (zlib not installed)
