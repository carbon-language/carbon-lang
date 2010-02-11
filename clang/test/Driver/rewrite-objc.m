// RUN: %clang -ccc-host-triple unknown -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: clang{{.*}}" "-rewrite-objc"

// RUN: not %clang -ccc-no-clang -ccc-host-triple unknown -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST1 %s
// TEST1: invalid output type 'rewritten-objc' for use with gcc

// RUN: not %clang -ccc-no-clang -ccc-host-triple i386-apple-darwin10 -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST2 %s
// TEST2: invalid output type 'rewritten-objc' for use with gcc
