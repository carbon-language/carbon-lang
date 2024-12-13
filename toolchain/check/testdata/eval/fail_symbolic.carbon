// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/fail_symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/fail_symbolic.carbon

// TODO: This should work.
fn G(N:! i32) {
  // CHECK:STDERR: fail_symbolic.carbon:[[@LINE+9]]:16: error: cannot implicitly convert from `i32` to `Core.Int(N)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var k: [i32; N];
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_symbolic.carbon:[[@LINE+6]]:16: note: type `i32` does not implement interface `Core.ImplicitAs(Core.Int(N))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var k: [i32; N];
  // CHECK:STDERR:                ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-opt/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:31:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
  // CHECK:STDERR:              ^~~~~~~~~~
  var k: [i32; N];
}

// CHECK:STDOUT: --- fail_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %N.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %iN [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.4, @impl.3(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.5) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.1, %Convert.5 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.4(%N.2) [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %ImplicitAs.type.4 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %N.patt.loc12_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc12_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc12_6.1, runtime_param<invalid> [symbolic = %N.patt.loc12_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc12_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc12_6.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_6.1: %i32) {
// CHECK:STDOUT:   %N.loc12_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc12_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc12_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc12_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.loc16: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc16 (constants.%N.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N.loc16) [symbolic = %Convert.type (constants.%Convert.type.5)]
// CHECK:STDOUT:   %Convert: @G.%Convert.type (%Convert.type.5) = struct_value () [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:   %Convert.bound.loc16_16.2: <bound method> = bound_method %N.loc12_6.2, %Convert [symbolic = %Convert.bound.loc16_16.2 (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_16.2: <specific function> = specific_function %Convert.bound.loc16_16.2, @Convert.4(%N.loc16) [symbolic = %Convert.specific_fn.loc16_16.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N.loc16 [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %require_complete.loc16_16.1: <witness> = require_complete_type @G.%iN (%iN) [symbolic = %require_complete.loc16_16.1 (constants.%require_complete.1)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %require_complete.loc16_16.2: <witness> = require_complete_type @G.%ImplicitAs.type (%ImplicitAs.type.4) [symbolic = %require_complete.loc16_16.2 (constants.%require_complete.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16: init type = call constants.%Int(%int_32.loc16) [template = constants.%i32]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc12_6.1 [symbolic = %N.loc12_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc16_11.1: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_11.2: type = converted %int.make_type_signed.loc16, %.loc16_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:     %Convert.bound.loc16_16.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound.loc16_16.2 (constants.%Convert.bound)]
// CHECK:STDOUT:     %Convert.specific_fn.loc16_16.1: <specific function> = specific_function %Convert.bound.loc16_16.1, @Convert.4(constants.%N.2) [symbolic = %Convert.specific_fn.loc16_16.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:     %.loc16_16.1: @G.%iN (%iN) = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.loc16_16.1(<error>) [template = <error>]
// CHECK:STDOUT:     %.loc16_16.2: Core.IntLiteral = value_of_initializer %int.convert_checked [template = <error>]
// CHECK:STDOUT:     %.loc16_16.3: Core.IntLiteral = converted %N.ref, %.loc16_16.2 [template = <error>]
// CHECK:STDOUT:     %array_type: type = array_type %.loc16_16.3, %i32 [template = <error>]
// CHECK:STDOUT:     %k.var: ref <error> = var k
// CHECK:STDOUT:     %k: ref <error> = bind_name k, %k.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.1) {
// CHECK:STDOUT:   %N.loc12_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc12_6.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
