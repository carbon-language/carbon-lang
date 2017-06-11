// REQUIRES: zlib

// RUN: %clang -### -fintegrated-as -Wa,-compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-_COMPRESS_DEBUG_SECTIONS %s
// RUN: %clang -### -fno-integrated-as -Wa,-compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-_COMPRESS_DEBUG_SECTIONS %s
// CHECK-_COMPRESS_DEBUG_SECTIONS: "-compress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-__COMPRESS_DEBUG_SECTIONS %s
// RUN: %clang -### -fno-integrated-as -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-__COMPRESS_DEBUG_SECTIONS %s
// CHECK-__COMPRESS_DEBUG_SECTIONS: "--compress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-POSNEG %s
// RUN: %clang -### -fno-integrated-as -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-POSNEG %s
// CHECK-POSNEG: "--compress-debug-sections"
// CHECK-POSNEG: "--nocompress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,-compress-debug-sections -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-MULTIPLE %s
// RUN: %clang -### -fno-integrated-as -Wa,-compress-debug-sections -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-MULTIPLE %s
// CHECK-MULTIPLE: "-compress-debug-sections"
// CHECK-MULTIPLE: "--compress-debug-sections"

