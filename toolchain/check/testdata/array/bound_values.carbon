// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
// CHECK:STDOUT:   %AddWith.type.d7d: type = facet_type <@AddWith, @AddWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.0ee: type = fn_type @AddWith.Op, @AddWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.568: <witness> = impl_witness imports.%AddWith.impl_witness_table.255 [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.d7d = facet_value Core.IntLiteral, (%AddWith.impl_witness.568) [concrete]
// CHECK:STDOUT:   %.e1a: type = fn_type_with_self_type %AddWith.Op.type.0ee, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op: %Core.IntLiteral.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.abd9 = import_ref Core//prelude/operators/arithmetic, loc95_57, unloaded
// CHECK:STDOUT:   %Core.import_ref.d10: %Core.IntLiteral.as.AddWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc96_42, loaded [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table.255 = impl_witness_table (%Core.import_ref.abd9, %Core.import_ref.d10), @Core.IntLiteral.as.AddWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_21: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1: %.e1a = impl_witness_access constants.%AddWith.impl_witness.568, element1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.AddWith.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1, %int_2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.AddWith.impl.Op.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.2: Core.IntLiteral = converted %Core.IntLiteral.as.AddWith.impl.Op.call, %.loc6_18.1 [concrete = constants.%int_3.1ba]
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
// CHECK:STDOUT:   %As.Convert.type.b94: type = fn_type @As.Convert, @As(%u32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.002: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.819: %UInt.as.ImplicitAs.impl.Convert.type.002 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.792: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.2f2(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c02: %Core.IntLiteral.as.As.impl.Convert.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.d2d: <witness> = impl_witness imports.%As.impl_witness_table.afe, @Core.IntLiteral.as.As.impl.2f2(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.70b: type = fn_type @Core.IntLiteral.as.As.impl.Convert.1, @Core.IntLiteral.as.As.impl.2f2(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.92a: %Core.IntLiteral.as.As.impl.Convert.type.70b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.ce0 = facet_value Core.IntLiteral, (%As.impl_witness.d2d) [concrete]
// CHECK:STDOUT:   %.68c: type = fn_type_with_self_type %As.Convert.type.b94, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.92a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.92a, @Core.IntLiteral.as.As.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.94d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.d14: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ad, @UInt.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.e06: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.47f: %UInt.as.ImplicitAs.impl.Convert.type.e06 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.84e: %ImplicitAs.type.2fd = facet_value %u32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.268: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.84e [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.d14, %UInt.as.ImplicitAs.impl.Convert.47f [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.47f, @UInt.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3a6: <bound method> = bound_method %int_3.d14, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.823: @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.002) = import_ref Core//prelude/types/uint, loc25_44, loaded [symbolic = @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.819)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ad = impl_witness_table (%Core.import_ref.823), @UInt.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.aef: @Core.IntLiteral.as.As.impl.2f2.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.792) = import_ref Core//prelude/types/uint, loc30_40, loaded [symbolic = @Core.IntLiteral.as.As.impl.2f2.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c02)]
// CHECK:STDOUT:   %As.impl_witness_table.afe = impl_witness_table (%Core.import_ref.aef), @Core.IntLiteral.as.As.impl.2f2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_24: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_32.loc6_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.1: %.68c = impl_witness_access constants.%As.impl_witness.d2d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.92a]
// CHECK:STDOUT:     %bound_method.loc6_18.1: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.2: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.94d]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u32 = call %bound_method.loc6_18.2(%int_3.loc6) [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.2: %u32 = converted %int_3.loc6, %.loc6_18.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %impl.elem0.loc6_18.2: %.268 = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.47f]
// CHECK:STDOUT:     %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @UInt.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.3a6]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
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
