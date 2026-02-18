// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.290: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(Core.IntLiteral, %ImplicitAs.facet.290) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet.290 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.dd4 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.17a: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.5d8 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0.loc11_21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc11_21.1: <bound method> = bound_method %int_3, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:     %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_21.2: <bound method> = bound_method %int_3, %specific_fn.loc11_21 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21: init %i32 = call %bound_method.loc11_21.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_21 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_21.2: %i32 = converted %int_3, %.loc11_21.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %F.call: init %i32 = call %F.ref(%.loc11_21.2) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc11_22: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:     %bound_method.loc11_22.1: <bound method> = bound_method %F.call, %impl.elem0.loc11_22 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc11_22: <specific function> = specific_function %impl.elem0.loc11_22, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_22.2: <bound method> = bound_method %F.call, %specific_fn.loc11_22 [concrete = constants.%bound_method.17a]
// CHECK:STDOUT:     %.loc11_22.1: %i32 = value_of_initializer %F.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_22.2: %i32 = converted %F.call, %.loc11_22.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc11_22.2(%.loc11_22.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_22.4: Core.IntLiteral = converted %F.call, %.loc11_22.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc11_22.4, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11_35.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc11_35.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc11_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc11_35.1: <specific function> = specific_function %impl.elem0.loc11_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc11_35.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1: init %i32 = call %bound_method.loc11_35.2(@__global_init.%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_35.1: init %i32 = converted @__global_init.%int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_35.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.3: ref %i32 = array_index %.loc11_35.2, %int_0
// CHECK:STDOUT:   %.loc11_35.4: init %i32 to %.loc11_35.3 = in_place_init %.loc11_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_35.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc11_35.3: <bound method> = bound_method @__global_init.%int_2, %impl.elem0.loc11_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc11_35.2: <specific function> = specific_function %impl.elem0.loc11_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.4: <bound method> = bound_method @__global_init.%int_2, %specific_fn.loc11_35.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2: init %i32 = call %bound_method.loc11_35.4(@__global_init.%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_35.5: init %i32 = converted @__global_init.%int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_35.6: ref %i32 = array_index %.loc11_35.2, %int_1
// CHECK:STDOUT:   %.loc11_35.7: init %i32 to %.loc11_35.6 = in_place_init %.loc11_35.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc11_35.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc11_35.5: <bound method> = bound_method @__global_init.%int_3, %impl.elem0.loc11_35.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc11_35.3: <specific function> = specific_function %impl.elem0.loc11_35.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.6: <bound method> = bound_method @__global_init.%int_3, %specific_fn.loc11_35.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3: init %i32 = call %bound_method.loc11_35.6(@__global_init.%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_35.8: init %i32 = converted @__global_init.%int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_35.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_35.9: ref %i32 = array_index %.loc11_35.2, %int_2
// CHECK:STDOUT:   %.loc11_35.10: init %i32 to %.loc11_35.9 = in_place_init %.loc11_35.8 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_35.11: init %array_type to %.loc11_35.2 = array_init (%.loc11_35.4, %.loc11_35.7, %.loc11_35.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.12: init %array_type = converted @__global_init.%.loc11, %.loc11_35.11 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_35.13: ref %array_type = temporary %.loc11_35.2, %.loc11_35.12
// CHECK:STDOUT:   %.loc11_35.14: %array_type = acquire_value %.loc11_35.13
// CHECK:STDOUT:   %n: %array_type = value_binding n, %.loc11_35.14
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
