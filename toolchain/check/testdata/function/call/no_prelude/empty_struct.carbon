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
// CHECK:STDOUT:   %Echo: type = fn_type @Echo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Echo = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT:   %struct.3: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Echo.decl: Echo = fn_decl @Echo [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_13.1: {} = struct_literal ()
// CHECK:STDOUT:     %.loc7_13.2: type = converted %.loc7_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %a.loc7_9.1: {} = param a
// CHECK:STDOUT:     @Echo.%a: {} = bind_name a, %a.loc7_9.1
// CHECK:STDOUT:     %.loc7_20.1: {} = struct_literal ()
// CHECK:STDOUT:     %.loc7_20.2: type = converted %.loc7_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Echo.%return: ref {} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {}
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
// CHECK:STDOUT:   %Echo.ref: Echo = name_ref Echo, file.%Echo.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_9: {} = struct_literal ()
// CHECK:STDOUT:   %struct: {} = struct_value () [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc12_7: {} = converted %.loc12_9, %struct [template = constants.%struct.3]
// CHECK:STDOUT:   %Echo.call: init {} = call %Echo.ref(%.loc12_7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
