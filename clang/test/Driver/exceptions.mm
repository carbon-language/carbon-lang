// RUN: %clang -target x86_64-apple-darwin11 -fno-exceptions %s -o - -### 2>&1 | \
// RUN:   FileCheck %s

CHECK-NOT: "-fobjc-exceptions"
CHECK-NOT: "-fcxx-exceptions"
CHECK-NOT: "-fexceptions"
