// REQUIRES: system-darwin

// On Darwin, -demangle is passed to the linker of HOST_LINK_VERSION
// is high enough. It is assumed to be high enough on systems where
// this test gets run.

// RUN: %clang -### %s 2>&1 | FileCheck %s
// CHECK: -demangle
