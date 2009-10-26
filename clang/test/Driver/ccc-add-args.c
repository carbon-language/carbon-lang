// RUN: env CCC_ADD_ARGS="-ccc-echo,-ccc-print-options,,-v" clang -### 2>&1 | FileCheck %s
// CHECK: Option 0 - Name: "-v", Values: {}
// CHECK: Option 1 - Name: "-###", Values: {}
