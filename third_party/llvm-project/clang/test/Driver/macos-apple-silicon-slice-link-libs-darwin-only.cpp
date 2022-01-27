// RUN: %clang -### -target arm64-apple-darwin -arch arm64 -mmacosx-version-min=10.7 %s 2>&1 | FileCheck -check-prefix=ARM64-10_7 %s
// RUN: %clang -### -target x86_64-apple-darwin10 -arch x86_64 -mmacosx-version-min=10.7 %s 2>&1 | FileCheck -check-prefix=x86_64-10_7 %s
// REQUIRES: system-darwin

// ARM64-10_7-NOT: -lcrt1.10.6.o
// x86_64-10_7:    -lcrt1.10.6.o
