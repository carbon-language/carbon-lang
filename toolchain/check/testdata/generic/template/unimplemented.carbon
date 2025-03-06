// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon

// --- todo_unimplemented.carbon

library "[[@TEST_NAME]]";

// Check that we get a reasonable diagnostic for an unimplemented operation on a template dependent expression.
fn F[template T:! type](x: T) -> i32 {
  // CHECK:STDERR: todo_unimplemented.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Mul` in type `<dependent type>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return x.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return x.n * 3;
}

// CHECK:STDOUT: --- todo_unimplemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Mul = %Core.Mul
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_15.1, runtime_param<none> [template = %T.patt.loc5_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc5_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc5_15.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc5_15.1: type = bind_symbolic_name T, 0, template, %T.param [template = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc5_15.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_15.1 [template = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc5_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_15.1: type) {
// CHECK:STDOUT:   %T.loc5_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc5_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc10_11.3: <instruction> = refine_type_action %x.ref, @F.%T.loc5_15.2 (%T) [template]
// CHECK:STDOUT:   %.loc10_11.4: <instruction> = access_member_action %.loc10_11.1, n [template]
// CHECK:STDOUT:   %.loc10_11.5: type = type_of_inst %.loc10_11.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @F.%T.loc5_15.2 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_11.1: @F.%T.loc5_15.2 (%T) = splice_inst %.loc10_11.3
// CHECK:STDOUT:     %.loc10_11.2: @F.%.loc10_11.5 (@F.%.loc10_11.5) = splice_inst %.loc10_11.4
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
