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

// CHECK:STDERR: fail_negative.carbon:[[@LINE+2]]:19-20: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:
var a: array(i32, -1);

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+2]]:19-38: error: array bound of 39999999999999999993 is too large [ArrayBoundTooLarge]
// CHECK:STDERR:
var a: array(i32, 39999999999999999993);

// --- fail_invalid_type_with_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+3]]:14: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+2]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
var b: array(1, 39999999999999999993);

// CHECK:STDOUT: --- addition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.195: type = facet_type <@AddWith, @AddWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.195 = facet_value Core.IntLiteral, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.3d9: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(Core.IntLiteral, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %.755: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.3d9, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op: %Core.IntLiteral.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.95c: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.AddWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.AddWith.impl.0df [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_21: type = splice_block %Array.loc6 [concrete = constants.%Array.95c] {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1: %.755 = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.AddWith.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1, %int_2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.AddWith.impl.Op.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.2: Core.IntLiteral = converted %Core.IntLiteral.as.AddWith.impl.Op.call, %.loc6_18.1 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array.loc6: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.95c]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.d0a: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.16e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.d83(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.81c: %Core.IntLiteral.as.As.impl.Convert.type.16e = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.f4f: <witness> = impl_witness imports.%As.impl_witness_table.43c, @Core.IntLiteral.as.As.impl.d83(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ea3: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.d83(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.0bc: %Core.IntLiteral.as.As.impl.Convert.type.ea3 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d0a = facet_value Core.IntLiteral, (%As.impl_witness.f4f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2f2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u32, %As.facet) [concrete]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %As.WithSelf.Convert.type.2f2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.0bc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.0bc, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.286: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.d32: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.802: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.d6a(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.7a2: %UInt.as.ImplicitAs.impl.Convert.type.802 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.adc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.7eb, @UInt.as.ImplicitAs.impl.d6a(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.311: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.d6a(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.9a6: %UInt.as.ImplicitAs.impl.Convert.type.311 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d02: %ImplicitAs.type.5be = facet_value %u32, (%ImplicitAs.impl_witness.adc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ae5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.d02) [concrete]
// CHECK:STDOUT:   %.4a3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ae5, %ImplicitAs.facet.d02 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.d32, %UInt.as.ImplicitAs.impl.Convert.9a6 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.9a6, @UInt.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.12e: <bound method> = bound_method %int_3.d32, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.95c: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d9c7: @Core.IntLiteral.as.As.impl.d83.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.16e) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.d83.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.81c)]
// CHECK:STDOUT:   %As.impl_witness_table.43c = impl_witness_table (%Core.import_ref.d9c7), @Core.IntLiteral.as.As.impl.d83 [concrete]
// CHECK:STDOUT:   %Core.import_ref.b7af: @UInt.as.ImplicitAs.impl.d6a.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.802) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.d6a.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.7a2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.7eb = impl_witness_table (%Core.import_ref.b7af), @UInt.as.ImplicitAs.impl.d6a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_24: type = splice_block %Array.loc6 [concrete = constants.%Array.95c] {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.1: %.a11 = impl_witness_access constants.%As.impl_witness.f4f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.0bc]
// CHECK:STDOUT:     %bound_method.loc6_18.1: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.2: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.286]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u32 = call %bound_method.loc6_18.2(%int_3.loc6) [concrete = constants.%int_3.d32]
// CHECK:STDOUT:     %.loc6_18.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_3.d32]
// CHECK:STDOUT:     %.loc6_18.2: %u32 = converted %int_3.loc6, %.loc6_18.1 [concrete = constants.%int_3.d32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.2: %.4a3 = impl_witness_access constants.%ImplicitAs.impl_witness.adc, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.9a6]
// CHECK:STDOUT:     %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.12e]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array.loc6: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.95c]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
