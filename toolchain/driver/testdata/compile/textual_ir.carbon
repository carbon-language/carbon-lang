// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/textual_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/textual_ir.carbon

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: --- textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %n.patt: %empty_tuple.type = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %empty_tuple.type = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_25.1: %tuple.type = tuple_literal (%.loc15_20, %.loc15_24)
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_20, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_25.3: type = converted %.loc15_24, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_25.4: type = converted %.loc15_25.1, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc15_12.1: type = splice_block %.loc15_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_12.3: type = converted %.loc15_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n.param_patt: %empty_tuple.type) -> %return.param_patt: %tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %.loc16_15.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_16.1: %tuple.type = tuple_literal (%n.ref, %.loc16_15.1)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %return, element0
// CHECK:STDOUT:   %.loc16_11: init %empty_tuple.type = tuple_init () to %tuple.elem0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.2: init %empty_tuple.type = converted %n.ref, %.loc16_11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %return, element1
// CHECK:STDOUT:   %.loc16_15.2: init %empty_tuple.type = tuple_init () to %tuple.elem1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.3: init %empty_tuple.type = converted %.loc16_15.1, %.loc16_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.4: init %tuple.type = tuple_init (%.loc16_16.2, %.loc16_16.3) to %return [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc16_17: init %tuple.type = converted %.loc16_16.1, %.loc16_16.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc16_17 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
