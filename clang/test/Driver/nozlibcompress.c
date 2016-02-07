// RUN: %clang -c %s -Wa,--compress-debug-sections 2>&1 | FileCheck %s
// RUN: %clang -c %s -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections 2>&1 | FileCheck --check-prefix=NOWARN %s
// REQUIRES: nozlib

// FIXME: This test hasn't run until r259976 made REQUIRES: zlib work -- and
// the test has been failing since. Figure out what's up and enable this.
// XFAIL: *

// CHECK: warning: cannot compress debug sections (zlib not installed)
// NOWARN-NOT: warning: cannot compress debug sections (zlib not installed)
