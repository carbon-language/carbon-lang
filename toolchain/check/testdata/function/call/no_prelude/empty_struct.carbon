// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/no_prelude/empty_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/no_prelude/empty_struct.carbon

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
// CHECK:STDOUT:   %Echo.type: type = fn_type @Echo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Echo: %Echo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Echo.decl: %Echo.type = fn_decl @Echo [template = constants.%Echo] {
// CHECK:STDOUT:     %.loc11_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_13.2: type = converted %.loc11_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %a.loc11_9.1: %.1 = param a, runtime_param0
// CHECK:STDOUT:     @Echo.%a: %.1 = bind_name a, %a.loc11_9.1
// CHECK:STDOUT:     %.loc11_20.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Echo.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: %.1) -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Echo.ref: %Echo.type = name_ref Echo, file.%Echo.decl [template = constants.%Echo]
// CHECK:STDOUT:   %.loc16_9: %.1 = struct_literal ()
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc16_7: %.1 = converted %.loc16_9, %struct [template = constants.%struct]
// CHECK:STDOUT:   %Echo.call: init %.1 = call %Echo.ref(%.loc16_7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
