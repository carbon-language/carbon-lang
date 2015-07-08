// RUN: clang-check "%s" -- -fasm-blocks -target x86_64-apple-darwin10 2>&1 | FileCheck -check-prefix=CHECK-X86 %s -allow-empty
// RUN: not clang-check "%s" -- -fasm-blocks -target powerpc-apple-darwin10 2>&1 | FileCheck -check-prefix=CHECK-PPC %s
// REQUIRES: x86-registered-target

void Break() {
  __asm { int 3 }
}

// clang-check should initialize the x86 target, so x86 should work.
// CHECK-X86-NOT: error: MS-style inline assembly is not available

// Test that the ordinary error is emitted on unsupported architectures.
// CHECK-PPC: error: Unsupported architecture 'powerpc' for MS-style inline assembly
