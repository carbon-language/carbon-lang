// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: {}) -> {} {
  return a;
}

fn Main() {
  Echo({});
}

// CHECK:STDOUT: --- empty_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Echo = %Echo, .Main = %Main} [template]
// CHECK:STDOUT:   %Echo: <function> = fn_decl @Echo [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: {}) -> {} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: {} = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Echo.ref: <function> = name_ref Echo, file.%Echo [template = file.%Echo]
// CHECK:STDOUT:   %.loc12_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc12_9.2: {} = struct_value () [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_9.3: {} = converted %.loc12_9.1, %.loc12_9.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_7: init {} = call %Echo.ref(%.loc12_9.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
