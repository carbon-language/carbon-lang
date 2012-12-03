// General blacklist usage.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%s %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST
// CHECK-BLACKLIST: -fsanitize-blacklist

// Ignore -fsanitize-blacklist flag if there is no -fsanitize flag.
// RUN: %clang -fsanitize-blacklist=%s %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE
// CHECK-NO-SANITIZE-NOT: -fsanitize-blacklist

// Flag -fno-sanitize-blacklist wins if it is specified later.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%s -fno-sanitize-blacklist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-BLACKLIST
// CHECK-NO-BLACKLIST-NOT: -fsanitize-blacklist

// Driver barks on unexisting blacklist files.
// RUN: %clang -fno-sanitize-blacklist -fsanitize-blacklist=unexisting.txt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SUCH-FILE
// CHECK-NO-SUCH-FILE: error: no such file or directory: 'unexisting.txt'

// PR12920
// XFAIL: cygwin,mingw32
