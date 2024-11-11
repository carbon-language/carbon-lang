// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/no_prelude/empty_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/no_prelude/empty_tuple.carbon

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  Echo(());
}

// CHECK:STDOUT: --- empty_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Echo.type: type = fn_type @Echo [template]
// CHECK:STDOUT:   %Echo: %Echo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Echo.decl: %Echo.type = fn_decl @Echo [template = constants.%Echo] {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %empty_tuple.type = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_13.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_13.2: type = converted %.loc11_13.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_20.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %a.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %a: %empty_tuple.type = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Echo.ref: %Echo.type = name_ref Echo, file.%Echo.decl [template = constants.%Echo]
// CHECK:STDOUT:   %.loc16_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_9.2: %empty_tuple.type = converted %.loc16_9.1, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %Echo.call: init %empty_tuple.type = call %Echo.ref(%.loc16_9.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
