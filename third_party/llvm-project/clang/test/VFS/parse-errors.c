// RUN: not %clang_cc1 -ivfsoverlay %S/Inputs/invalid-yaml.yaml -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: invalid virtual filesystem overlay file

// RUN: not %clang_cc1 -ivfsoverlay %S/Inputs/missing-key.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-MISSING-TYPE %s
// CHECK-MISSING-TYPE: missing key 'type'
// CHECK-MISSING-TYPE: invalid virtual filesystem overlay file

// RUN: not %clang_cc1 -ivfsoverlay %S/Inputs/unknown-key.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-UNKNOWN-KEY %s
// CHECK-UNKNOWN-KEY: unknown key
// CHECK-UNKNOWN-KEY: invalid virtual filesystem overlay file

// RUN: not %clang_cc1 -ivfsoverlay %S/Inputs/unknown-value.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-UNKNOWN-VALUE %s
// CHECK-UNKNOWN-VALUE: expected boolean value
// CHECK-UNKNOWN-VALUE: invalid virtual filesystem overlay file
