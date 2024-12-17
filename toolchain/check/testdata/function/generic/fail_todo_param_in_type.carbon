// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon

// CHECK:STDERR: fail_todo_param_in_type.carbon:[[@LINE+9]]:24: error: cannot implicitly convert from `i32` to `Core.Int(N)` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn F(N:! i32, a: [i32; N]*);
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_todo_param_in_type.carbon:[[@LINE+6]]:24: note: type `i32` does not implement interface `Core.ImplicitAs(Core.Int(N))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn F(N:! i32, a: [i32; N]*);
// CHECK:STDERR:                        ^
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:31:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:              ^~~~~~~~~~
fn F(N:! i32, a: [i32; N]*);

// CHECK:STDOUT: --- fail_todo_param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %N.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN.1: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%iN.1)> [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %ImplicitAs.type.4 [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %iN.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.4, @impl.3(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.9: <witness> = impl_witness (%Convert.5), @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N.1, %Convert.5 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.4(%N.2) [symbolic]
// CHECK:STDOUT:   %iN.2: type = int_type signed, %N.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %N.1, %Convert.15 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.4(%N.1) [symbolic]
// CHECK:STDOUT:   %require_complete.7: <witness> = require_complete_type %iN.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%iN.2)> [symbolic]
// CHECK:STDOUT:   %require_complete.8: <witness> = require_complete_type %ImplicitAs.type.6 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc20_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc20_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc20_6.1, runtime_param<invalid> [symbolic = %N.patt.loc20_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc20_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc20_10: init type = call constants.%Int(%int_32.loc20_10) [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_10.1: type = value_of_initializer %int.make_type_signed.loc20_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_10.2: type = converted %int.make_type_signed.loc20_10, %.loc20_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc20_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc20_19: init type = call constants.%Int(%int_32.loc20_19) [template = constants.%i32]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc20_6.1 [symbolic = %N.loc20_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc20_19.1: type = value_of_initializer %int.make_type_signed.loc20_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_19.2: type = converted %int.make_type_signed.loc20_19, %.loc20_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:     %Convert.bound.loc20_24.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound.loc20_24.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:     %Convert.specific_fn.loc20_24.1: <specific function> = specific_function %Convert.bound.loc20_24.1, @Convert.4(constants.%N.2) [symbolic = %Convert.specific_fn.loc20_24.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:     %.loc20_24.1: @F.%iN (%iN.1) = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.loc20_24.1(<error>) [template = <error>]
// CHECK:STDOUT:     %.loc20_24.2: Core.IntLiteral = value_of_initializer %int.convert_checked [template = <error>]
// CHECK:STDOUT:     %.loc20_24.3: Core.IntLiteral = converted %N.ref, %.loc20_24.2 [template = <error>]
// CHECK:STDOUT:     %array_type: type = array_type %.loc20_24.3, %i32 [template = <error>]
// CHECK:STDOUT:     %ptr: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc20_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc20_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc20_6.1: %i32) {
// CHECK:STDOUT:   %N.loc20_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc20_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc20_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc20_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %N.loc20_24: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc20_24 (constants.%N.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N.loc20_24) [symbolic = %Convert.type (constants.%Convert.type.5)]
// CHECK:STDOUT:   %Convert: @F.%Convert.type (%Convert.type.5) = struct_value () [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:   %Convert.bound.loc20_24.2: <bound method> = bound_method %N.loc20_6.2, %Convert [symbolic = %Convert.bound.loc20_24.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn.loc20_24.2: <specific function> = specific_function %Convert.bound.loc20_24.2, @Convert.4(%N.loc20_24) [symbolic = %Convert.specific_fn.loc20_24.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N.loc20_24 [symbolic = %iN (constants.%iN.1)]
// CHECK:STDOUT:   %require_complete.loc20_24.1: <witness> = require_complete_type @F.%iN (%iN.1) [symbolic = %require_complete.loc20_24.1 (constants.%require_complete.2)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %require_complete.loc20_24.2: <witness> = require_complete_type @F.%ImplicitAs.type (%ImplicitAs.type.4) [symbolic = %require_complete.loc20_24.2 (constants.%require_complete.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32, %a.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.1) {
// CHECK:STDOUT:   %N.loc20_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc20_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.loc20_24 => constants.%N.1
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.15
// CHECK:STDOUT:   %Convert => constants.%Convert.15
// CHECK:STDOUT:   %Convert.bound.loc20_24.2 => constants.%Convert.bound.2
// CHECK:STDOUT:   %Convert.specific_fn.loc20_24.2 => constants.%Convert.specific_fn.2
// CHECK:STDOUT:   %iN => constants.%iN.2
// CHECK:STDOUT:   %require_complete.loc20_24.1 => constants.%require_complete.7
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %require_complete.loc20_24.2 => constants.%require_complete.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
