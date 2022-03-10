// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-macos10.13 -fuse-ld=lld \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=0 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s
// RUN: %clang -target x86_64-apple-macos10.13 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: env SDKROOT=%S/Inputs/MacOSX10.14.sdk %clang \
// RUN:   -target x86_64-apple-macos10.13.0.1 -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s

// RUN: %clang -target arm64-apple-macos10.13 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_OLD %s
// RUN: %clang -target arm64-apple-macos10.13 -fuse-ld=lld \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=400 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s
// RUN: %clang -target arm64-apple-macos10.13 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s
// RUN: %clang -target arm64-apple-darwin19 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s
// RUN: %clang -target arm64-apple-macos11.1 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW_1 %s

// LINKER-OLD: "-macosx_version_min" "10.13.0"
// LINKER-NEW: "-platform_version" "macos" "10.13.0" "10.14"

// ARM64_NEW: "-platform_version" "macos" "11.0.0" "10.14"
// ARM64_NEW_1: "-platform_version" "macos" "11.1.0" "10.14"
// ARM64_OLD: "-macosx_version_min" "11.0.0"

// RUN: %clang -target x86_64-apple-macos10.13 -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSDK %s
// RUN: %clang -target x86_64-apple-darwin17 -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSDK %s
// NOSDK: "-platform_version" "macos" "10.13.0" "10.13.0"
