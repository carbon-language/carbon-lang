// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+3]]:18: ERROR: Cannot implicitly convert from `type` to `i32`.
// CHECK:STDERR: var x: (i32, ) = (i32, );
// CHECK:STDERR:                  ^~~~~~~
var x: (i32, ) = (i32, );

// CHECK:STDOUT: --- fail_type_assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_14.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc10_14.2: type = converted %.loc10_14.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref (i32,) = var x
// CHECK:STDOUT:   %x: ref (i32,) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_24: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
