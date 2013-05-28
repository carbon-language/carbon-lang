// REQUIRES: shell-preserves-root

// RUN: touch %t.o

// RUN: %clang -target i386-unknown-linux --dyld-prefix /foo -### %t.o 2>&1 | FileCheck --check-prefix=CHECK-32 %s
// CHECK-32: "-dynamic-linker" "/foo/lib/ld-linux.so.2"

// RUN: %clang -target x86_64-unknown-linux --dyld-prefix /foo -### %t.o 2>&1 | FileCheck --check-prefix=CHECK-64 %s
// CHECK-64: "-dynamic-linker" "/foo/lib64/ld-linux-x86-64.so.2"
