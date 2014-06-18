// RUN: %clang -target armv7-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target armv7-eabi -### -mlong-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LONG-CALLS

// RUN: %clang -target armv7-eabi -### -mlong-calls -mno-long-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-LONG-CALLS

// CHECK-DEFAULT-NOT: "-backend-option" "-arm-long-calls"

// CHECK-LONG-CALLS: "-backend-option" "-arm-long-calls"

// CHECK-NO-LONG-CALLS-NOT: "-backend-option" "-arm-long-calls"

