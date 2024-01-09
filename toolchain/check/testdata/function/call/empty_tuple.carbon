// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  Echo(());
}

// CHECK:STDOUT: --- empty_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7: type = tuple_type ()
// CHECK:STDOUT:   %.loc12: () = tuple_value ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Echo = %Echo, .Main = %Main}
// CHECK:STDOUT:   %Echo: <function> = fn_decl @Echo
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: () = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Echo.ref: <function> = name_ref Echo, file.%Echo
// CHECK:STDOUT:   %.loc12_3: <function> = reify_constant %Echo.ref, file.%Echo
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: () = tuple_value ()
// CHECK:STDOUT:   %.loc12_9.3: () = reify_constant %.loc12_9.2, constants.%.loc12
// CHECK:STDOUT:   %.loc12_9.4: () = converted %.loc12_9.1, %.loc12_9.3
// CHECK:STDOUT:   %.loc12_7: init () = call file.%Echo(%.loc12_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
