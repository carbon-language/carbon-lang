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
// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+4]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var b: array(1, 39999999999999999993);
// CHECK:STDERR:              ^
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
// CHECK:STDOUT:   %AddWith.type.832: type = facet_type <@AddWith, @AddWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.832 = facet_value Core.IntLiteral, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.a21: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(Core.IntLiteral, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.a21, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op: %Core.IntLiteral.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.022: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.AddWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.AddWith.impl.8f4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_21: type = splice_block %Array.loc6 [concrete = constants.%Array.022] {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1: %.fbc = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.AddWith.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1, %int_2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.AddWith.impl.Op.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.2: Core.IntLiteral = converted %Core.IntLiteral.as.AddWith.impl.Op.call, %.loc6_18.1 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array.loc6: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.022]
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
// CHECK:STDOUT:   %As.type.349: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.873: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.31e(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2ed: %Core.IntLiteral.as.As.impl.Convert.type.873 = struct_value () [symbolic]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.f0e: <witness> = impl_witness imports.%As.impl_witness_table.c5f, @Core.IntLiteral.as.As.impl.31e(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e10: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.31e(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.74e: %Core.IntLiteral.as.As.impl.Convert.type.e10 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.349 = facet_value Core.IntLiteral, (%As.impl_witness.f0e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.1ab: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%u32, %As.facet) [concrete]
// CHECK:STDOUT:   %.746: type = fn_type_with_self_type %As.WithSelf.Convert.type.1ab, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.74e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.74e, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.08a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f2: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.c0a: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.43d(%From.fe9) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.1c1: %UInt.as.ImplicitAs.impl.Convert.type.c0a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cf7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0c0, @UInt.as.ImplicitAs.impl.43d(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.9f6: type = fn_type @UInt.as.ImplicitAs.impl.Convert.1, @UInt.as.ImplicitAs.impl.43d(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.5e4: %UInt.as.ImplicitAs.impl.Convert.type.9f6 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.32f: %ImplicitAs.type.7cb = facet_value %u32, (%ImplicitAs.impl_witness.cf7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.85b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.32f) [concrete]
// CHECK:STDOUT:   %.3c9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.85b, %ImplicitAs.facet.32f [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.9f2, %UInt.as.ImplicitAs.impl.Convert.5e4 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.5e4, @UInt.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.581: <bound method> = bound_method %int_3.9f2, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Array.022: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d09: @Core.IntLiteral.as.As.impl.31e.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.873) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.31e.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2ed)]
// CHECK:STDOUT:   %As.impl_witness_table.c5f = impl_witness_table (%Core.import_ref.d09), @Core.IntLiteral.as.As.impl.31e [concrete]
// CHECK:STDOUT:   %Core.import_ref.a5a8f: @UInt.as.ImplicitAs.impl.43d.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.c0a) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.43d.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.1c1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0c0 = impl_witness_table (%Core.import_ref.a5a8f), @UInt.as.ImplicitAs.impl.43d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_24: type = splice_block %Array.loc6 [concrete = constants.%Array.022] {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.1: %.746 = impl_witness_access constants.%As.impl_witness.f0e, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.74e]
// CHECK:STDOUT:     %bound_method.loc6_18.1: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.2: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.08a]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u32 = call %bound_method.loc6_18.2(%int_3.loc6) [concrete = constants.%int_3.9f2]
// CHECK:STDOUT:     %.loc6_18.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_3.9f2]
// CHECK:STDOUT:     %.loc6_18.2: %u32 = converted %int_3.loc6, %.loc6_18.1 [concrete = constants.%int_3.9f2]
// CHECK:STDOUT:     %impl.elem0.loc6_18.2: %.3c9 = impl_witness_access constants.%ImplicitAs.impl_witness.cf7, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.5e4]
// CHECK:STDOUT:     %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @UInt.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.581]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array.loc6: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.022]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
