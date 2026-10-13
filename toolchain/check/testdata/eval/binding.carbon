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
  // CHECK:STDERR: fail_global.carbon:[[@LINE+3]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  // CHECK:STDERR:   return a;
  // CHECK:STDERR:          ^
  return a;
}

// CHECK:STDERR: fail_global.carbon:[[@LINE+4]]:19: note: in call to G here [InCallToEvalFn]
// CHECK:STDERR: let n: array(i32, G()) = (1, 2, 3);
// CHECK:STDERR:                   ^~~
// CHECK:STDERR:
let n: array(i32, G()) = (1, 2, 3);

// --- fail_runtime_value.carbon

library "[[@TEST_NAME]]";

fn F() -> i32;

let a: i32 = F();

eval fn G() -> i32 {
  // CHECK:STDERR: fail_runtime_value.carbon:[[@LINE+3]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  // CHECK:STDERR:   return a;
  // CHECK:STDERR:          ^
  return a;
}

// CHECK:STDERR: fail_runtime_value.carbon:[[@LINE+4]]:19: note: in call to G here [InCallToEvalFn]
// CHECK:STDERR: let n: array(i32, G()) = (1, 2, 3);
// CHECK:STDERR:                   ^~~
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
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.0ab: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.fab: %Int.as.ImplicitAs.impl.Convert.type.0ab = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.604: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.604) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet.604 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.690: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.0aa: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.34b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9ad, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f74: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.412: %Int.as.ImplicitAs.impl.Convert.type.f74 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3e: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.34b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7e7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.a3e) [concrete]
// CHECK:STDOUT:   %.53e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7e7, %ImplicitAs.facet.a3e [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.0aa, %Int.as.ImplicitAs.impl.Convert.412 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.412, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3c3: <bound method> = bound_method %int_3.0aa, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.022: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %pattern_type.9bf: type = pattern_type %Array.022 [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.9bf = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.022 = tuple_value (%int_1.874, %int_2.92b, %int_3.0aa) [concrete]
// CHECK:STDOUT:   %.d46: ref %Array.022 = temporary invalid, %array [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.03c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.0ab) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.fab)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9ad = impl_witness_table (%Core.import_ref.03c), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0.loc11_35.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc11_35.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc11_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc11_35.1: <specific function> = specific_function %impl.elem0.loc11_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc11_35.1 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1: init %i32 = call %bound_method.loc11_35.2(@__global_init.%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc11_35.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc11_35.2: ref %Array.022 = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.3: ref %i32 = array_index %.loc11_35.2, %int_0
// CHECK:STDOUT:   %.loc11_35.4: init %i32 to %.loc11_35.3 = in_place_init %.loc11_35.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %impl.elem0.loc11_35.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc11_35.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc11_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc11_35.2: <specific function> = specific_function %impl.elem0.loc11_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc11_35.2 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2: init %i32 = call %bound_method.loc11_35.4(@__global_init.%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc11_35.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_35.6: ref %i32 = array_index %.loc11_35.2, %int_1
// CHECK:STDOUT:   %.loc11_35.7: init %i32 to %.loc11_35.6 = in_place_init %.loc11_35.5 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %impl.elem0.loc11_35.3: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc11_35.5: <bound method> = bound_method @__global_init.%int_3, %impl.elem0.loc11_35.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b]
// CHECK:STDOUT:   %specific_fn.loc11_35.3: <specific function> = specific_function %impl.elem0.loc11_35.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.6: <bound method> = bound_method @__global_init.%int_3, %specific_fn.loc11_35.3 [concrete = constants.%bound_method.690]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3: init %i32 = call %bound_method.loc11_35.6(@__global_init.%int_3) [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %.loc11_35.8: init %i32 = converted @__global_init.%int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_35.9: ref %i32 = array_index %.loc11_35.2, %int_2
// CHECK:STDOUT:   %.loc11_35.10: init %i32 to %.loc11_35.9 = in_place_init %.loc11_35.8 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %.loc11_35.11: init %Array.022 to %.loc11_35.2 = array_init (%.loc11_35.4, %.loc11_35.7, %.loc11_35.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.12: init %Array.022 = converted @__global_init.%.loc11, %.loc11_35.11 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.13: ref %Array.022 = temporary %.loc11_35.2, %.loc11_35.12 [concrete = constants.%.d46]
// CHECK:STDOUT:   %.loc11_35.14: %Array.022 = acquire_value %.loc11_35.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_23: type = splice_block %Array [concrete = constants.%Array.022] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0.loc11_21: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc11_21.1: <bound method> = bound_method %int_3, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b]
// CHECK:STDOUT:     %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_21.2: <bound method> = bound_method %int_3, %specific_fn.loc11_21 [concrete = constants.%bound_method.690]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21: init %i32 = call %bound_method.loc11_21.2(%int_3) [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %.loc11_21.2: %i32 = converted %int_3, %.loc11_21.1 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %F.call: init %i32 = call %F.ref(%.loc11_21.2) [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %impl.elem0.loc11_22: %.53e = impl_witness_access constants.%ImplicitAs.impl_witness.34b, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.412]
// CHECK:STDOUT:     %bound_method.loc11_22.1: <bound method> = bound_method %F.call, %impl.elem0.loc11_22 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_22: <specific function> = specific_function %impl.elem0.loc11_22, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_22.2: <bound method> = bound_method %F.call, %specific_fn.loc11_22 [concrete = constants.%bound_method.3c3]
// CHECK:STDOUT:     %.loc11_22.1: %i32 = value_of_initializer %F.call [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %.loc11_22.2: %i32 = converted %F.call, %.loc11_22.1 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_22.2(%.loc11_22.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.4: Core.IntLiteral = converted %F.call, %.loc11_22.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.022]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %Array.022 = wrapper_binding n, %.loc11_35.14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.9bf = value_binding_pattern n [concrete = constants.%n.patt]
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
