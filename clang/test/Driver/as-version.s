// Test version information.

// UNSUPPORTED: -zos
// RUN: %clang -Wa,--version -c -fintegrated-as %s -o /dev/null \
// RUN:   | FileCheck --check-prefix=IAS %s
// IAS: clang version
