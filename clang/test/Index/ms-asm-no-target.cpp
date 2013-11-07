// XFAIL: aarch64, arm, mips, hexagon, powerpc, sparc
// REQUIRES: x86-registered-target
// RUN: c-index-test -test-load-source all -fasm-blocks -Wno-microsoft %s 2>&1 | FileCheck %s

// Test that we diagnose when the application hasn't initialized LLVM targets
// supporting the MS-style inline asm parser.

void Break() {
  __asm { int 3 }
}
// CHECK: error: MS-style inline assembly is not available
