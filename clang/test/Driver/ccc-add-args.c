// RUN: env CCC_ADD_ARGS="-ccc-echo,-ccc-print-options,,-v" clang -### 2> %t &&
// RUN: grep -F 'Option 0 - Name: "-v", Values: {}' %t &&
// RUN: grep -F 'Option 1 - Name: "-###", Values: {}' %t
