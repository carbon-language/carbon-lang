// RUN: %clang -### -c -save-temps -integrated-as %s 2>&1 | FileCheck %s

// CHECK: cc1as
// CHECK: -mrelax-all

// RUN: %clang -### -c -integrated-as -Wa,-compress-debug-sections -Wno-missing-debug-compression %s 2>&1 | FileCheck --check-prefix=COMPRESS_DEBUG_QUIET %s
// COMPRESS_DEBUG_QUIET-NOT: warning: DWARF compression is not implemented
// COMPRESS_DEBUG_QUIET-NOT: warning: argument unused during compilation
// COMPRESS_DEBUG_QUIET: -cc1
