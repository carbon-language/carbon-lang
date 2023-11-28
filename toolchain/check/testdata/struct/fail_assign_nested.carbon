// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:19: ERROR: Missing value for field `a` in struct initialization.
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                   ^~~~~~~~~
var x: {.a: {}} = {.b = {}};

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_14.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_14.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc10_15: type = struct_type {.a: ()}
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.b: {}}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_assign_nested.carbon" {
// CHECK:STDOUT:   %.loc10_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_14.2: type = converted %.loc10_14.1, constants.%.loc10_14.1
// CHECK:STDOUT:   %.loc10_15: type = struct_type {.a: {}}
// CHECK:STDOUT:   %x.var: ref {.a: {}} = var x
// CHECK:STDOUT:   %x: ref {.a: {}} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_26: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_27: {.b: {}} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
