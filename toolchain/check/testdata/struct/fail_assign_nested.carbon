// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:27: ERROR: Mismatched names for field 1 in struct initialization: source has field name `b`, destination has field name `a`.
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                           ^
var x: {.a: {}} = {.b = {}};

// CHECK:STDOUT: file "fail_assign_nested.carbon" {
// CHECK:STDOUT:   %.loc10_14.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_14.2: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_15.1: type = struct_type {.a: {}}
// CHECK:STDOUT:   %.loc10_14.3: type = tuple_type ()
// CHECK:STDOUT:   %.loc10_15.2: type = struct_type {.a: ()}
// CHECK:STDOUT:   %x: ref {.a: {}} = var "x"
// CHECK:STDOUT:   %.loc10_26: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_27.1: type = struct_type {.b: {}}
// CHECK:STDOUT:   %.loc10_27.2: {.b: {}} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
