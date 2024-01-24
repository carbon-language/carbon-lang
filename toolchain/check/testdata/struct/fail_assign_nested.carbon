// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:19: ERROR: Missing value for field `a` in struct initialization.
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                   ^~~~~~~~~
var x: {.a: {}} = {.b = {}};

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: {}} [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: ()} [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: {}} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_14.2: type = converted %.loc10_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_15: type = struct_type {.a: {}} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref {.a: {}} = var x
// CHECK:STDOUT:   %x: ref {.a: {}} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_26: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_27: {.b: {}} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
