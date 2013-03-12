// RUN: %clang -### -Wlarge-by-value-copy %s 2>&1 | FileCheck -check-prefix=LARGE_VALUE_COPY_DEFAULT %s
// LARGE_VALUE_COPY_DEFAULT: -Wlarge-by-value-copy=64
// RUN: %clang -### -Wlarge-by-value-copy=128 %s 2>&1 | FileCheck -check-prefix=LARGE_VALUE_COPY_JOINED %s
// LARGE_VALUE_COPY_JOINED: -Wlarge-by-value-copy=128

// FIXME: Remove this together with -Warc-abi once an Xcode is released that doesn't pass this flag.
// RUN: %clang -### -Warc-abi -Wno-arc-abi %s 2>&1 | FileCheck -check-prefix=ARCABI %s
// ARCABI-NOT: unknown warning option '-Warc-abi'
// ARCABI-NOT: unknown warning option '-Wno-arc-abi'

// Check that -isysroot warns on nonexistent paths.
// RUN: %clang -### -c -target i386-apple-darwin10 -isysroot /FOO %s 2>&1 | FileCheck --check-prefix=CHECK-ISYSROOT %s
// CHECK-ISYSROOT: warning: no such sysroot directory: '{{([A-Za-z]:.*)?}}/FOO'
