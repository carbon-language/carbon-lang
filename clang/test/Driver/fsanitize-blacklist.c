// General blacklist usage.

// PR12920
// REQUIRES: clang-driver, shell

// Make sure we don't match the -NOT lines with the linker invocation.
// Delimiters match the start of the cc1 and the start of the linker lines
// for fragile tests.
// DELIMITERS: {{^ *"}}

// RUN: echo "fun:foo" > %t.good
// RUN: echo "fun:bar" > %t.second
// RUN: echo "badline" > %t.bad

// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.good -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST
// CHECK-BLACKLIST: -fsanitize-blacklist={{.*}}.good
// CHECK-BLACKLIST: -fsanitize-blacklist={{.*}}.second

// Ignore -fsanitize-blacklist flag if there is no -fsanitize flag.
// RUN: %clang -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE --check-prefix=DELIMITERS
// CHECK-NO-SANITIZE-NOT: -fsanitize-blacklist

// Flag -fno-sanitize-blacklist wins if it is specified later.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.good -fno-sanitize-blacklist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-BLACKLIST --check-prefix=DELIMITERS
// CHECK-NO-BLACKLIST-NOT: -fsanitize-blacklist

// Driver barks on unexisting blacklist files.
// RUN: %clang -fno-sanitize-blacklist -fsanitize-blacklist=unexisting.txt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SUCH-FILE
// CHECK-NO-SUCH-FILE: error: no such file or directory: 'unexisting.txt'

// Driver properly reports malformed blacklist files.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.second -fsanitize-blacklist=%t.bad -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-BLACKLIST
// CHECK-BAD-BLACKLIST: error: malformed sanitizer blacklist: 'error parsing file '{{.*}}.bad': malformed line 1: 'badline''

// -fno-sanitize-blacklist disables all blacklists specified earlier.
// RUN: %clang -fsanitize=address -fsanitize-blacklist=%t.good -fno-sanitize-blacklist -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-FIRST-DISABLED
// CHECK-ONLY_FIRST-DISABLED-NOT: good
// CHECK-ONLY-FIRST-DISABLED: -fsanitize-blacklist={{.*}}.second
// CHECK-ONLY_FIRST-DISABLED-NOT: good

// DELIMITERS: {{^ *"}}
