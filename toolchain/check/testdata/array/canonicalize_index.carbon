// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/canonicalize_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/canonicalize_index.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn ConvertToU32(a: i32) -> u32 = "int.convert_checked";

// CHECK:STDERR: canonicalize_index.carbon:[[@LINE+4]]:14: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var a: [i32; Add(1, 2)] = (1, 2, 3);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;
// CHECK:STDERR: canonicalize_index.carbon:[[@LINE+9]]:14: error: cannot implicitly convert from `u32` to `Core.UInt(N)` [ImplicitAsConversionFailure]
// CHECK:STDERR: let c: [i32; ConvertToU32(3)]* = &a;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~
// CHECK:STDERR: canonicalize_index.carbon:[[@LINE+6]]:14: note: type `u32` does not implement interface `Core.ImplicitAs(Core.UInt(N))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let c: [i32; ConvertToU32(3)]* = &a;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~
// CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-opt/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/operators/as.carbon:35:14: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:              ^~~~~~~~~~
let c: [i32; ConvertToU32(3)]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [template]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (%Convert.3), @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.3(%int_3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %u32 = call %ConvertToU32(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (%Convert.7), @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int.convert_checked.4, %Convert.7 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.5(%N) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ConvertToU32 = %ConvertToU32.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
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
// CHECK:STDOUT:   %ConvertToU32.decl: %ConvertToU32.type = fn_decl @ConvertToU32 [template = constants.%ConvertToU32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32.loc12_20) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_20.2: type = converted %int.make_type_signed, %.loc12_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32.loc12_28) [template = constants.%u32]
// CHECK:STDOUT:     %.loc12_28.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc12_28.2: type = converted %int.make_type_unsigned, %.loc12_28.1 [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int_32.loc18) [template = constants.%i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc18_18: %Convert.type.2 = impl_witness_access constants.%impl_witness.1, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_18: <bound method> = bound_method %int_1, %impl.elem0.loc18_18 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_18: <specific function> = specific_function %Convert.bound.loc18_18, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_18: init %iN = call %Convert.specific_fn.loc18_18(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_18.1: %i32 = value_of_initializer %int.convert_checked.loc18_18 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_18.2: %i32 = converted %int_1, %.loc18_18.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc18_21: %Convert.type.2 = impl_witness_access constants.%impl_witness.1, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_21: <bound method> = bound_method %int_2, %impl.elem0.loc18_21 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_21: <specific function> = specific_function %Convert.bound.loc18_21, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc18_21: init %iN = call %Convert.specific_fn.loc18_21(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc18_21.1: %i32 = value_of_initializer %int.convert_checked.loc18_21 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc18_21.2: %i32 = converted %int_2, %.loc18_21.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %Add.ref(%.loc18_18.2, %.loc18_21.2)
// CHECK:STDOUT:   %.loc18_9.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_9.2: type = converted %int.make_type_signed.loc18, %.loc18_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%int_32.loc19) [template = constants.%i32]
// CHECK:STDOUT:   %int_3.loc19: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc19_9.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_9.2: type = converted %int.make_type_signed.loc19, %.loc19_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc19: type = array_type %int_3.loc19, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr.loc19: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %int_32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%int_32.loc29) [template = constants.%i32]
// CHECK:STDOUT:   %ConvertToU32.ref: %ConvertToU32.type = name_ref ConvertToU32, %ConvertToU32.decl [template = constants.%ConvertToU32]
// CHECK:STDOUT:   %int_3.loc29: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc29_27: %Convert.type.2 = impl_witness_access constants.%impl_witness.1, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc29_27: <bound method> = bound_method %int_3.loc29, %impl.elem0.loc29_27 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_27: <specific function> = specific_function %Convert.bound.loc29_27, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc29_27: init %iN = call %Convert.specific_fn.loc29_27(%int_3.loc29) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc29_27.1: %i32 = value_of_initializer %int.convert_checked.loc29_27 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc29_27.2: %i32 = converted %int_3.loc29, %.loc29_27.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc29_28.1: init %u32 = call %ConvertToU32.ref(%.loc29_27.2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc29_9.1: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_9.2: type = converted %int.make_type_signed.loc29, %.loc29_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc29_28: %Convert.type.6 = impl_witness_access constants.%impl_witness.2, element0 [symbolic = constants.%Convert.7]
// CHECK:STDOUT:   %Convert.bound.loc29_28: <bound method> = bound_method %int.convert_checked.loc29_28.1, %impl.elem0.loc29_28 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_28: <specific function> = specific_function %Convert.bound.loc29_28, @Convert.5(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %.loc29_28.1: %uN = converted %int.convert_checked.loc29_28.1, <error> [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc29_28.2: init Core.IntLiteral = call %Convert.specific_fn.loc29_28(<error>) [template = <error>]
// CHECK:STDOUT:   %.loc29_28.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc29_28.2 [template = <error>]
// CHECK:STDOUT:   %.loc29_28.3: Core.IntLiteral = converted %int.convert_checked.loc29_28.1, %.loc29_28.2 [template = <error>]
// CHECK:STDOUT:   %array_type.loc29: type = array_type %.loc29_28.3, %i32 [template = <error>]
// CHECK:STDOUT:   %ptr.loc29: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToU32(%a.param_patt: %i32) -> %u32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc18: %tuple.type = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %a.ref.loc19: ref <error> = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc19: <error> = addr_of %a.ref.loc19
// CHECK:STDOUT:   %b: %ptr = bind_name b, <error>
// CHECK:STDOUT:   %a.ref.loc29: ref <error> = name_ref a, file.%a
// CHECK:STDOUT:   %addr.loc29: <error> = addr_of %a.ref.loc29
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
