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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_return_var_no_returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_return_var_no_returned_var.carbon

fn Procedure() -> {} {
  // CHECK:STDERR: fail_return_var_no_returned_var.carbon:[[@LINE+4]]:3: error: `return var;` with no `returned var` in scope [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  return var;
}

// CHECK:STDOUT: --- fail_return_var_no_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Procedure.type: type = fn_type @Procedure [concrete]
// CHECK:STDOUT:   %Procedure: %Procedure.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Procedure = %Procedure.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Procedure.decl: %Procedure.type = fn_decl @Procedure [concrete = constants.%Procedure] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_20.2: type = converted %.loc15_20.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
