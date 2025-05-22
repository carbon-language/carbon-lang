// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/bound_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/bound_values.carbon

// --- addition.carbon

library "[[@TEST_NAME]]";

var a:
//@dump-sem-ir-begin
    array(i32, 1 + 2)
//@dump-sem-ir-end
    = (1, 2, 3);
let b: array(i32, 3)* = &a;

// --- unsigned.carbon

library "[[@TEST_NAME]]";

var a:
//@dump-sem-ir-begin
    array(i32, 3 as u32)
//@dump-sem-ir-end
    = (1, 2, 3);
let b: array(i32, 3)* = &a;

// --- fail_negative.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_negative.carbon:[[@LINE+4]]:19: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR: var a: array(i32, -1);
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var a: array(i32, -1);

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:19: error: array bound of 39999999999999999993 is too large [ArrayBoundTooLarge]
// CHECK:STDERR: var a: array(i32, 39999999999999999993);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, 39999999999999999993);

// --- fail_invalid_type_with_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+7]]:14: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var b: array(1, 39999999999999999993);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+4]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: array(1, 39999999999999999993);
// CHECK:STDERR:              ^
// CHECK:STDERR:
var b: array(1, 39999999999999999993);

// CHECK:STDOUT: --- addition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Add.impl_witness.b91: <witness> = impl_witness imports.%Add.impl_witness_table.5a4 [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value Core.IntLiteral, (%Add.impl_witness.b91) [concrete]
// CHECK:STDOUT:   %.066: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.type.f28: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.d0f: %Op.type.f28 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.5b8, %Op.d0f [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.3e4: %Op.type.f28 = import_ref Core//prelude/operators/arithmetic, loc78_42, loaded [concrete = constants.%Op.d0f]
// CHECK:STDOUT:   %Add.impl_witness_table.5a4 = impl_witness_table (%Core.import_ref.3e4), @impl.9e0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_21: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.066 = impl_witness_access constants.%Add.impl_witness.b91, element0 [concrete = constants.%Op.d0f]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:     %int.sadd: init Core.IntLiteral = call %bound_method(%int_1, %int_2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = value_of_initializer %int.sadd [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.2: Core.IntLiteral = converted %int.sadd, %.loc6_18.1 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc6: type = array_type %.loc6_18.2, %i32.loc6 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.ce0: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %Convert.type.b94: type = fn_type @Convert.1, @As(%u32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.002: type = fn_type @Convert.4, @impl.750(%From) [symbolic]
// CHECK:STDOUT:   %Convert.819: %Convert.type.002 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.type.792: type = fn_type @Convert.5, @impl.2f2(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.c02: %Convert.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.d2d: <witness> = impl_witness imports.%As.impl_witness_table.afe, @impl.2f2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.70b: type = fn_type @Convert.5, @impl.2f2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.92a: %Convert.type.70b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.ce0 = facet_value Core.IntLiteral, (%As.impl_witness.d2d) [concrete]
// CHECK:STDOUT:   %.68c: type = fn_type_with_self_type %Convert.type.b94, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.7d9: <bound method> = bound_method %int_3.1ba, %Convert.92a [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.4cf: <specific function> = specific_function %Convert.92a, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.94d: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn.4cf [concrete]
// CHECK:STDOUT:   %int_3.d14: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ad, @impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e06: type = fn_type @Convert.4, @impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.47f: %Convert.type.e06 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.84e: %ImplicitAs.type.2fd = facet_value %u32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.268: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.84e [concrete]
// CHECK:STDOUT:   %Convert.bound.258: <bound method> = bound_method %int_3.d14, %Convert.47f [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.2eb: <specific function> = specific_function %Convert.47f, @Convert.4(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3a6: <bound method> = bound_method %int_3.d14, %Convert.specific_fn.2eb [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.823: @impl.750.%Convert.type (%Convert.type.002) = import_ref Core//prelude/types/uint, loc24_44, loaded [symbolic = @impl.750.%Convert (constants.%Convert.819)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ad = impl_witness_table (%Core.import_ref.823), @impl.750 [concrete]
// CHECK:STDOUT:   %Core.import_ref.aef: @impl.2f2.%Convert.type (%Convert.type.792) = import_ref Core//prelude/types/uint, loc29_40, loaded [symbolic = @impl.2f2.%Convert (constants.%Convert.c02)]
// CHECK:STDOUT:   %As.impl_witness_table.afe = impl_witness_table (%Core.import_ref.aef), @impl.2f2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_24: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_32.loc6_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.1: %.68c = impl_witness_access constants.%As.impl_witness.d2d, element0 [concrete = constants.%Convert.92a]
// CHECK:STDOUT:     %bound_method.loc6_18.1: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_18.1 [concrete = constants.%Convert.bound.7d9]
// CHECK:STDOUT:     %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.4cf]
// CHECK:STDOUT:     %bound_method.loc6_18.2: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.94d]
// CHECK:STDOUT:     %int.convert_checked.loc6_18.1: init %u32 = call %bound_method.loc6_18.2(%int_3.loc6) [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.1: %u32 = value_of_initializer %int.convert_checked.loc6_18.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.2: %u32 = converted %int_3.loc6, %.loc6_18.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %impl.elem0.loc6_18.2: %.268 = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.47f]
// CHECK:STDOUT:     %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%Convert.bound.258]
// CHECK:STDOUT:     %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @Convert.4(constants.%int_32) [concrete = constants.%Convert.specific_fn.2eb]
// CHECK:STDOUT:     %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.3a6]
// CHECK:STDOUT:     %int.convert_checked.loc6_18.2: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_18.2 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc6: type = array_type %.loc6_18.4, %i32.loc6 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
