// RUN: not clang-check "%s" -- -fasm-blocks -target x86_64-apple-darwin10 2>&1 | FileCheck -check-prefix=CHECK-X86 %s
// RUN: not clang-check "%s" -- -fasm-blocks -target powerpc-apple-darwin10 2>&1 | FileCheck -check-prefix=CHECK-PPC %s

// Test that we diagnose instead of crashing when the application hasn't
// initialized LLVM targets supporting the MS-style inline asm parser.
// Also test that the ordinary error is emitted on unsupported architectures.

void Break() {
  __asm { int 3 }
}

// CHECK-X86: error: MS-style inline assembly is not available
// CHECK-PPC: error: Unsupported architecture 'powerpc' for MS-style inline assembly
