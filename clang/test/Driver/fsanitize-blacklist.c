// General blacklist usage.

// PR12920
// REQUIRES: clang-driver, shell

// RUN: echo "fun:foo" > %t.good
// RUN: echo "badline" > %t.bad

// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST
// CHECK-BLACKLIST: -fsanitize-blacklist

// Ignore -fsanitize-blacklist flag if there is no -fsanitize flag.
// RUN: %clang -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE
// CHECK-NO-SANITIZE-NOT: -fsanitize-blacklist

// Flag -fno-sanitize-blacklist wins if it is specified later.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.good -fno-sanitize-blacklist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-BLACKLIST
// CHECK-NO-BLACKLIST-NOT: -fsanitize-blacklist

// Driver barks on unexisting blacklist files.
// RUN: %clang -fno-sanitize-blacklist -fsanitize-blacklist=unexisting.txt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SUCH-FILE
// CHECK-NO-SUCH-FILE: error: no such file or directory: 'unexisting.txt'

// Driver properly reports malformed blacklist files.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.bad %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-BLACKLIST
// CHECK-BAD-BLACKLIST: error: malformed sanitizer blacklist
