// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/call.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDERR: call.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; Add(1, 2)];
// CHECK:STDERR:                ^~~~~~~~~
var arr: [i32; Add(1, 2)];

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_11: init type = call constants.%Int(%int_32.loc11_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_signed.loc11_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_signed.loc11_11, %.loc11_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_19: init type = call constants.%Int(%int_32.loc11_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_signed.loc11_19, %.loc11_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_27: init type = call constants.%Int(%int_32.loc11_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_signed.loc11_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_signed.loc11_27, %.loc11_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc13_20: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_20: <bound method> = bound_method %int_1, %impl.elem0.loc13_20 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20: <specific function> = specific_function %Convert.bound.loc13_20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_20: init %iN = call %Convert.specific_fn.loc13_20(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_1, %.loc13_20.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc13_23: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_23: <bound method> = bound_method %int_2, %impl.elem0.loc13_23 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_23: <specific function> = specific_function %Convert.bound.loc13_23, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_23: init %iN = call %Convert.specific_fn.loc13_23(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_23.1: %i32 = value_of_initializer %int.convert_checked.loc13_23 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_23.2: %i32 = converted %int_2, %.loc13_23.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %Add.ref(%.loc13_20.2, %.loc13_23.2)
// CHECK:STDOUT:   %.loc13_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_11.2: type = converted %int.make_type_signed, %.loc13_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
