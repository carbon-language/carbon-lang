// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/empty_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/empty_tuple.carbon

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  Echo(());
}

// CHECK:STDOUT: --- empty_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Echo.type: type = fn_type @Echo [concrete]
// CHECK:STDOUT:   %Echo: %Echo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Echo.decl: %Echo.type = fn_decl @Echo [concrete = constants.%Echo] {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_20.2: type = converted %.loc15_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %a.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc15_13.1: type = splice_block %.loc15_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_13.3: type = converted %.loc15_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %empty_tuple.type = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a.param: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %.loc16_10: init %empty_tuple.type = tuple_init () to %return [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_11: init %empty_tuple.type = converted %a.ref, %.loc16_10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc16_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Echo.ref: %Echo.type = name_ref Echo, file.%Echo.decl [concrete = constants.%Echo]
// CHECK:STDOUT:   %.loc20_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_9.2: %empty_tuple.type = converted %.loc20_9.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Echo.call: init %empty_tuple.type = call %Echo.ref(%.loc20_9.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
