// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/binding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/binding.carbon

// --- local.carbon

library "[[@TEST_NAME]]";

eval fn F(a: i32) -> i32 {
  let b: i32 = a;
  let c: i32 = b;
  return c;
}

//@dump-sem-ir-begin
let n: array(i32, F(3)) = (1, 2, 3);
//@dump-sem-ir-end

// --- fail_global.carbon

library "[[@TEST_NAME]]";

// This is a runtime binding, even though it has a compile-time-constant value,
// so is not accessible from within G.
let a: i32 = 3;

eval fn G() -> i32 {
  // CHECK:STDERR: fail_global.carbon:[[@LINE+1]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  return a;
}

// CHECK:STDERR: fail_global.carbon:[[@LINE+2]]:19-21: note: in call to G here [InCallToEvalFn]
// CHECK:STDERR:
let n: array(i32, G()) = (1, 2, 3);

// --- fail_runtime_value.carbon

library "[[@TEST_NAME]]";

fn F() -> i32;

let a: i32 = F();

eval fn G() -> i32 {
  // CHECK:STDERR: fail_runtime_value.carbon:[[@LINE+1]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  return a;
}

// CHECK:STDERR: fail_runtime_value.carbon:[[@LINE+2]]:19-21: note: in call to G here [InCallToEvalFn]
// CHECK:STDERR:
let n: array(i32, G()) = (1, 2, 3);

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dfd: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.914: %Int.as.ImplicitAs.impl.Convert.type.dfd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.323: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b7c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f39: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.779: %Int.as.ImplicitAs.impl.Convert.type.f39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0aa: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.323) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4c7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.0aa) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4c7, %ImplicitAs.facet.0aa [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.9f3, %Int.as.ImplicitAs.impl.Convert.779 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.779, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.746: <bound method> = bound_method %int_3.9f3, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.95c: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %pattern_type.d4e: type = pattern_type %Array.95c [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d4e = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.95c = tuple_value (%int_1.68a, %int_2.2e0, %int_3.9f3) [concrete]
// CHECK:STDOUT:   %.c07: ref %Array.95c = temporary invalid, %array [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ed0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.dfd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.914)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b7c = impl_witness_table (%Core.import_ref.ed0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0.loc11_35.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_35.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc11_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc11_35.1: <specific function> = specific_function %impl.elem0.loc11_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc11_35.1 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1: init %i32 = call %bound_method.loc11_35.2(@__global_init.%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_35.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_35.2: ref %Array.95c = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.3: ref %i32 = array_index %.loc11_35.2, %int_0
// CHECK:STDOUT:   %.loc11_35.4: init %i32 to %.loc11_35.3 = in_place_init %.loc11_35.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc11_35.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_35.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc11_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc11_35.2: <specific function> = specific_function %impl.elem0.loc11_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc11_35.2 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2: init %i32 = call %bound_method.loc11_35.4(@__global_init.%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc11_35.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_35.6: ref %i32 = array_index %.loc11_35.2, %int_1
// CHECK:STDOUT:   %.loc11_35.7: init %i32 to %.loc11_35.6 = in_place_init %.loc11_35.5 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc11_35.3: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_35.5: <bound method> = bound_method @__global_init.%int_3, %impl.elem0.loc11_35.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc11_35.3: <specific function> = specific_function %impl.elem0.loc11_35.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.6: <bound method> = bound_method @__global_init.%int_3, %specific_fn.loc11_35.3 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3: init %i32 = call %bound_method.loc11_35.6(@__global_init.%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc11_35.8: init %i32 = converted @__global_init.%int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_35.9: ref %i32 = array_index %.loc11_35.2, %int_2
// CHECK:STDOUT:   %.loc11_35.10: init %i32 to %.loc11_35.9 = in_place_init %.loc11_35.8 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc11_35.11: init %Array.95c to %.loc11_35.2 = array_init (%.loc11_35.4, %.loc11_35.7, %.loc11_35.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.12: init %Array.95c = converted @__global_init.%.loc11, %.loc11_35.11 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.13: ref %Array.95c = temporary %.loc11_35.2, %.loc11_35.12 [concrete = constants.%.c07]
// CHECK:STDOUT:   %.loc11_35.14: %Array.95c = acquire_value %.loc11_35.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_23: type = splice_block %Array [concrete = constants.%Array.95c] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0.loc11_21: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc11_21.1: <bound method> = bound_method %int_3, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:     %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_21.2: <bound method> = bound_method %int_3, %specific_fn.loc11_21 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21: init %i32 = call %bound_method.loc11_21.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc11_21.2: %i32 = converted %int_3, %.loc11_21.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %F.call: init %i32 = call %F.ref(%.loc11_21.2) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %impl.elem0.loc11_22: %.ce7 = impl_witness_access constants.%ImplicitAs.impl_witness.323, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.779]
// CHECK:STDOUT:     %bound_method.loc11_22.1: <bound method> = bound_method %F.call, %impl.elem0.loc11_22 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_22: <specific function> = specific_function %impl.elem0.loc11_22, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_22.2: <bound method> = bound_method %F.call, %specific_fn.loc11_22 [concrete = constants.%bound_method.746]
// CHECK:STDOUT:     %.loc11_22.1: %i32 = value_of_initializer %F.call [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc11_22.2: %i32 = converted %F.call, %.loc11_22.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_22.2(%.loc11_22.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.4: Core.IntLiteral = converted %F.call, %.loc11_22.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.95c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %Array.95c = wrapper_binding n, %.loc11_35.14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.d4e = value_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc11: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
