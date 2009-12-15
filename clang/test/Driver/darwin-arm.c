// RUN: %clang -ccc-host-triple i386-apple-darwin9 -arch arm -print-search-dirs | FileCheck %s

// Check that we look in the relative libexec directory.
// CHECK: {{programs: =.*/../libexec:}}
