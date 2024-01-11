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
// CHECK:STDOUT:   %.loc7_13.1: type = struct_type {}, const
// CHECK:STDOUT:   %.loc7_13.2: type = tuple_type (), const
// CHECK:STDOUT:   %.loc12: {} = struct_value (), const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Echo = %Echo, .Main = %Main}
// CHECK:STDOUT:   %Echo: <function> = fn_decl @Echo, const
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main, const
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
// CHECK:STDOUT:   %Echo.ref: <function> = name_ref Echo, file.%Echo, const = file.%Echo
// CHECK:STDOUT:   %.loc12_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc12_9.2: {} = struct_value (), const = constants.%.loc12
// CHECK:STDOUT:   %.loc12_9.3: {} = converted %.loc12_9.1, %.loc12_9.2
// CHECK:STDOUT:   %.loc12_7: init {} = call %Echo.ref(%.loc12_9.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
