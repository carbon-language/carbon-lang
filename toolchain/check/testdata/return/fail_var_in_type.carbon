// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: type = i32;
// CHECK:STDERR: fail_var_in_type.carbon:[[@LINE+3]]:13: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
fn Six() -> x { return 6; }

// CHECK:STDOUT: --- fail_var_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Six = %Six
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref type = var x
// CHECK:STDOUT:   %x: ref type = bind_name x, %x.var
// CHECK:STDOUT:   %Six: <function> = fn_decl @Six [template] {
// CHECK:STDOUT:     %x.ref: ref type = name_ref x, %x
// CHECK:STDOUT:     %.loc11: type = bind_value %x.ref
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 6 [template = constants.%.1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%x.var, i32
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
