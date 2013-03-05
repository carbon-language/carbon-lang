// Test -targat and -mabi.

// RUN: %clang -target arm-none-none-gnu -mabi=aapcs %s -c -S -o %t.s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WARN1 %s
//CHECK-WARN1: warning: unused environment 'gnu'

// RUN: %clang -target arm-none-none-gnueabi -mabi=apcs-gnu %s -c -S -o %t.s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WARN2 %s
// CHECK-WARN2: warning: unused environment 'gnueabi'

// RUN: %clang -target arm-none-none-gnueabi -mabi=aapcs-gnu %s -c -S -o %t.s 2>&1 \
// RUN:   | FileCheck %s
// CHECK-NOT: warning: unused environment

// RUN: %clang -target arm-none-none-gnu -mabi=apcs-gnu %s -c -S -o %t.s 2>&1 \
// RUN:   | FileCheck %s
// CHECK-NOT: warning: unused environment
