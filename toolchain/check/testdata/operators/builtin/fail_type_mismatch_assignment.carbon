// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch_assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch_assignment.carbon

fn Main() {
  var a: i32 = 3;
  // CHECK:STDERR: fail_type_mismatch_assignment.carbon:[[@LINE+7]]:3: error: cannot implicitly convert from `f64` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   a = 5.6;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_type_mismatch_assignment.carbon:[[@LINE+4]]:3: note: type `f64` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   a = 5.6;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  a = 5.6;
}

// CHECK:STDOUT: --- fail_type_mismatch_assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [template]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_3.1ba, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %float: f64 = float_literal 5.6000000000000005 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %.loc12_3.1: %i32 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_3, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %specific_fn(%int_3) [template = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_3.2: init %i32 = converted %int_3, %int.convert_checked [template = constants.%int_3.822]
// CHECK:STDOUT:   assign %a.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %float: f64 = float_literal 5.6000000000000005 [template = constants.%float]
// CHECK:STDOUT:   %.loc20: %i32 = converted %float, <error> [template = <error>]
// CHECK:STDOUT:   assign %a.ref, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
