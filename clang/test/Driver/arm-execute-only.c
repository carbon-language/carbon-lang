// RUN: not %clang -c -target thumbv6m-eabi -mexecute-only %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-EXECUTE-ONLY-NOT-SUPPORTED %s
// CHECK-EXECUTE-ONLY-NOT-SUPPORTED: error: execute only is not supported for the thumbv6m sub-architecture

// RUN: not %clang -target armv8m.main-eabi -mexecute-only -mno-movt %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY-NO-MOVT
// CHECK-EXECUTE-ONLY-NO-MOVT: error: option '-mexecute-only' cannot be specified with '-mno-movt'

// RUN: not %clang -target armv8m.main-eabi -mexecute-only -mlong-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY-LONG-CALLS
// CHECK-EXECUTE-ONLY-LONG-CALLS: error: option '-mexecute-only' cannot be specified with '-mlong-calls'

// RUN: %clang -target armv7m-eabi -x assembler -mexecute-only %s -c -### 2>&1 \
// RUN:    | FileCheck %s --check-prefix=CHECK-NO-EXECUTE-ONLY-ASM
// CHECK-NO-EXECUTE-ONLY-ASM: warning: argument unused during compilation: '-mexecute-only'

// -mpure-code flag for GCC compatibility
// RUN: not %clang -c -target thumbv6m-eabi -mpure-code %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-EXECUTE-ONLY-NOT-SUPPORTED %s

// RUN: not %clang -target armv8m.main-eabi -mpure-code -mno-movt %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-PURE-CODE-NO-MOVT
// CHECK-PURE-CODE-NO-MOVT: error: option '-mpure-code' cannot be specified with '-mno-movt'

// RUN: not %clang -target armv8m.main-eabi -mpure-code -mlong-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-PURE-CODE-LONG-CALLS
// CHECK-PURE-CODE-LONG-CALLS: error: option '-mpure-code' cannot be specified with '-mlong-calls'
