// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: (i32,) = (0,);
var b_ref: (((i32,), i32), (i32, i32)) = (((0,), 1), (2, 3));

class C(X:! (i32, i32)) {}

fn F() -> C((1, 2));

// --- implicit.impl.carbon

impl package Implicit;

var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;
var c: C((1, 2)) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var c_bad: C((1, 2, 3)) = F();
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-5]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon:7:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(X:! (i32, i32)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C((1, 2, 3)) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C((1, 2))` to `C((3, 4))` [ImplicitAsConversionFailure]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C((1, 2))` does not implement interface `Core.ImplicitAs(C((3, 4)))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.0e4e: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %int_0.5c61: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.985f: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b412: <bound method> = bound_method %int_0.5c61, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c350: <specific function> = specific_function %Convert.bound.b412, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61a: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple.2c2d: %tuple.type.0e4e = tuple_value (%int_0.f61a) [template]
// CHECK:STDOUT:   %tuple.type.24b0: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.0e1e: type = tuple_type (%tuple.type.0e4e, %i32) [template]
// CHECK:STDOUT:   %tuple.type.4719: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.b19c: type = tuple_type (%tuple.type.0e1e, %tuple.type.4719) [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.1866: type = tuple_type (%tuple.type.985f, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1baa: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.f949: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.d47b: type = tuple_type (%tuple.type.1866, %tuple.type.f949) [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.20cf: %tuple.type.0e1e = tuple_value (%tuple.2c2d, %int_1.c605) [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db0: <bound method> = bound_method %int_3.1baa, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4567: <specific function> = specific_function %Convert.bound.0db0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25ba: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple.44c6: %tuple.type.4719 = tuple_value (%int_2.1667, %int_3.25ba) [template]
// CHECK:STDOUT:   %tuple.1255: %tuple.type.b19c = tuple_value (%tuple.20cf, %tuple.44c6) [template]
// CHECK:STDOUT:   %X: %tuple.type.4719 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.4719 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.617b: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.58e6: %tuple.type.4719 = tuple_value (%int_1.c605, %int_2.1667) [template]
// CHECK:STDOUT:   %C.6141: type = class_type @C, @C(%tuple.58e6) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.0e4e = var a_ref
// CHECK:STDOUT:   %a_ref: ref %tuple.type.0e4e = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.b19c = var b_ref
// CHECK:STDOUT:   %b_ref: ref %tuple.type.b19c = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %X.patt.loc7_9.1: %tuple.type.4719 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: %tuple.type.4719 = value_param_pattern %X.patt.loc7_9.1, runtime_param<invalid> [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.param: %tuple.type.4719 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_22.1: type = splice_block %.loc7_22.3 [template = constants.%tuple.type.4719] {
// CHECK:STDOUT:       %int_32.loc7_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %int_32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %.loc7_22.2: %tuple.type.24b0 = tuple_literal (%i32.loc7_14, %i32.loc7_19)
// CHECK:STDOUT:       %.loc7_22.3: type = converted %.loc7_22.2, constants.%tuple.type.4719 [template = constants.%tuple.type.4719]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc7_9.1: %tuple.type.4719 = bind_symbolic_name X, 0, %X.param [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.6141 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.6141 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:     %.loc9_18.1: %tuple.type.f949 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_18.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:     %Convert.bound.loc9_18.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_18.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_18.1: <specific function> = specific_function %Convert.bound.loc9_18.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.1: init %i32 = call %Convert.specific_fn.loc9_18.1(%int_1) [template = constants.%int_1.c605]
// CHECK:STDOUT:     %.loc9_18.2: %i32 = value_of_initializer %int.convert_checked.loc9_18.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:     %.loc9_18.3: %i32 = converted %int_1, %.loc9_18.2 [template = constants.%int_1.c605]
// CHECK:STDOUT:     %impl.elem0.loc9_18.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:     %Convert.bound.loc9_18.2: <bound method> = bound_method %int_2, %impl.elem0.loc9_18.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_18.2: <specific function> = specific_function %Convert.bound.loc9_18.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.2: init %i32 = call %Convert.specific_fn.loc9_18.2(%int_2) [template = constants.%int_2.1667]
// CHECK:STDOUT:     %.loc9_18.4: %i32 = value_of_initializer %int.convert_checked.loc9_18.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:     %.loc9_18.5: %i32 = converted %int_2, %.loc9_18.4 [template = constants.%int_2.1667]
// CHECK:STDOUT:     %tuple: %tuple.type.4719 = tuple_value (%.loc9_18.3, %.loc9_18.5) [template = constants.%tuple.58e6]
// CHECK:STDOUT:     %.loc9_19: %tuple.type.4719 = converted %.loc9_18.1, %tuple [template = constants.%tuple.58e6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.58e6) [template = constants.%C.6141]
// CHECK:STDOUT:     %return.param: ref %C.6141 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.6141 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc7_9.1: %tuple.type.4719) {
// CHECK:STDOUT:   %X.loc7_9.2: %tuple.type.4719 = bind_symbolic_name X, 0 [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc7_9.2: %tuple.type.4719 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3571]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.617b
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.6141;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %.loc4_24.1: %tuple.type.985f = tuple_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %Convert.specific_fn.loc4(%int_0.loc4) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc4_24.2: init %i32 = converted %int_0.loc4, %int.convert_checked.loc4 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc4_24.3: init %tuple.type.0e4e = tuple_init (%.loc4_24.2) to file.%a_ref.var [template = constants.%tuple.2c2d]
// CHECK:STDOUT:   %.loc4_25: init %tuple.type.0e4e = converted %.loc4_24.1, %.loc4_24.3 [template = constants.%tuple.2c2d]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_25
// CHECK:STDOUT:   %int_0.loc5: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %.loc5_47.1: %tuple.type.985f = tuple_literal (%int_0.loc5)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc5_51.1: %tuple.type.1866 = tuple_literal (%.loc5_47.1, %int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %.loc5_59.1: %tuple.type.f949 = tuple_literal (%int_2, %int_3)
// CHECK:STDOUT:   %.loc5_60.1: %tuple.type.d47b = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %impl.elem0.loc5_47: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc5_47: <bound method> = bound_method %int_0.loc5, %impl.elem0.loc5_47 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_47: <specific function> = specific_function %Convert.bound.loc5_47, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc5_47: init %i32 = call %Convert.specific_fn.loc5_47(%int_0.loc5) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc5_47.2: init %i32 = converted %int_0.loc5, %int.convert_checked.loc5_47 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %tuple.elem0.loc5_60: ref %tuple.type.0e1e = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_51: ref %tuple.type.0e4e = tuple_access %tuple.elem0.loc5_60, element0
// CHECK:STDOUT:   %.loc5_47.3: init %tuple.type.0e4e = tuple_init (%.loc5_47.2) to %tuple.elem0.loc5_51 [template = constants.%tuple.2c2d]
// CHECK:STDOUT:   %.loc5_51.2: init %tuple.type.0e4e = converted %.loc5_47.1, %.loc5_47.3 [template = constants.%tuple.2c2d]
// CHECK:STDOUT:   %.loc5_51.3: init %tuple.type.0e4e = initialize_from %.loc5_51.2 to %tuple.elem0.loc5_51 [template = constants.%tuple.2c2d]
// CHECK:STDOUT:   %impl.elem0.loc5_51: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc5_51: <bound method> = bound_method %int_1, %impl.elem0.loc5_51 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_51: <specific function> = specific_function %Convert.bound.loc5_51, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc5_51: init %i32 = call %Convert.specific_fn.loc5_51(%int_1) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc5_51.4: init %i32 = converted %int_1, %int.convert_checked.loc5_51 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %tuple.elem1.loc5_51: ref %i32 = tuple_access %tuple.elem0.loc5_60, element1
// CHECK:STDOUT:   %.loc5_51.5: init %i32 = initialize_from %.loc5_51.4 to %tuple.elem1.loc5_51 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc5_51.6: init %tuple.type.0e1e = tuple_init (%.loc5_51.3, %.loc5_51.5) to %tuple.elem0.loc5_60 [template = constants.%tuple.20cf]
// CHECK:STDOUT:   %.loc5_60.2: init %tuple.type.0e1e = converted %.loc5_51.1, %.loc5_51.6 [template = constants.%tuple.20cf]
// CHECK:STDOUT:   %impl.elem0.loc5_59.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc5_59.1: <bound method> = bound_method %int_2, %impl.elem0.loc5_59.1 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_59.1: <specific function> = specific_function %Convert.bound.loc5_59.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.1: init %i32 = call %Convert.specific_fn.loc5_59.1(%int_2) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc5_59.2: init %i32 = converted %int_2, %int.convert_checked.loc5_59.1 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %tuple.elem1.loc5_60: ref %tuple.type.4719 = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element0
// CHECK:STDOUT:   %.loc5_59.3: init %i32 = initialize_from %.loc5_59.2 to %tuple.elem0.loc5_59 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %impl.elem0.loc5_59.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc5_59.2: <bound method> = bound_method %int_3, %impl.elem0.loc5_59.2 [template = constants.%Convert.bound.0db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_59.2: <specific function> = specific_function %Convert.bound.loc5_59.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4567]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.2: init %i32 = call %Convert.specific_fn.loc5_59.2(%int_3) [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc5_59.4: init %i32 = converted %int_3, %int.convert_checked.loc5_59.2 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %tuple.elem1.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element1
// CHECK:STDOUT:   %.loc5_59.5: init %i32 = initialize_from %.loc5_59.4 to %tuple.elem1.loc5_59 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc5_59.6: init %tuple.type.4719 = tuple_init (%.loc5_59.3, %.loc5_59.5) to %tuple.elem1.loc5_60 [template = constants.%tuple.44c6]
// CHECK:STDOUT:   %.loc5_60.3: init %tuple.type.4719 = converted %.loc5_59.1, %.loc5_59.6 [template = constants.%tuple.44c6]
// CHECK:STDOUT:   %.loc5_60.4: init %tuple.type.b19c = tuple_init (%.loc5_60.2, %.loc5_60.3) to file.%b_ref.var [template = constants.%tuple.1255]
// CHECK:STDOUT:   %.loc5_61: init %tuple.type.b19c = converted %.loc5_60.1, %.loc5_60.4 [template = constants.%tuple.1255]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5_61
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.58e6) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%tuple.58e6
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%tuple.58e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.5907: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.7c47: type = tuple_type (%tuple.type.5907, %i32) [template]
// CHECK:STDOUT:   %tuple.type.e076: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.fced: type = tuple_type (%tuple.type.7c47, %tuple.type.e076) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ed97: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.e076 = tuple_value (%int_1.0593, %int_2.ed97) [template]
// CHECK:STDOUT:   %C.fd9a: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.c6d3: ref %tuple.type.5907 = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %import_ref.1fe7: ref %tuple.type.fced = import_ref Implicit//default, b_ref, loaded
// CHECK:STDOUT:   %import_ref.c9bd: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.0dce: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d695
// CHECK:STDOUT:     .ImplicitAs = %import_ref.edd8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.2e04 = import_ref Implicit//default, inst1052 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.c6d3
// CHECK:STDOUT:     .b_ref = imports.%import_ref.1fe7
// CHECK:STDOUT:     .C = imports.%import_ref.c9bd
// CHECK:STDOUT:     .F = imports.%import_ref.0dce
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %tuple.type.5907 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.5907 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %tuple.type.fced = var b
// CHECK:STDOUT:   %b: ref %tuple.type.fced = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %C.fd9a = var c
// CHECK:STDOUT:   %c: ref %C.fd9a = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.e076) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2e04
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.fd9a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.5907 = name_ref a_ref, imports.%import_ref.c6d3
// CHECK:STDOUT:   %tuple.elem0.loc4: ref %i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_17.1: %i32 = bind_value %tuple.elem0.loc4
// CHECK:STDOUT:   %.loc4_17.2: init %tuple.type.5907 = tuple_init (%.loc4_17.1) to file.%a.var
// CHECK:STDOUT:   %.loc4_22: init %tuple.type.5907 = converted %a_ref.ref, %.loc4_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc4_22
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.fced = name_ref b_ref, imports.%import_ref.1fe7
// CHECK:STDOUT:   %tuple.elem0.loc5_38.1: ref %tuple.type.7c47 = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.2: ref %tuple.type.5907 = tuple_access %tuple.elem0.loc5_38.1, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.3: ref %i32 = tuple_access %tuple.elem0.loc5_38.2, element0
// CHECK:STDOUT:   %.loc5_38.1: %i32 = bind_value %tuple.elem0.loc5_38.3
// CHECK:STDOUT:   %tuple.elem0.loc5_38.4: ref %tuple.type.7c47 = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.5: ref %tuple.type.5907 = tuple_access %tuple.elem0.loc5_38.4, element0
// CHECK:STDOUT:   %.loc5_38.2: init %tuple.type.5907 = tuple_init (%.loc5_38.1) to %tuple.elem0.loc5_38.5
// CHECK:STDOUT:   %.loc5_38.3: init %tuple.type.5907 = converted %tuple.elem0.loc5_38.2, %.loc5_38.2
// CHECK:STDOUT:   %.loc5_38.4: init %tuple.type.5907 = initialize_from %.loc5_38.3 to %tuple.elem0.loc5_38.5
// CHECK:STDOUT:   %tuple.elem1.loc5_38.1: ref %i32 = tuple_access %tuple.elem0.loc5_38.1, element1
// CHECK:STDOUT:   %.loc5_38.5: %i32 = bind_value %tuple.elem1.loc5_38.1
// CHECK:STDOUT:   %tuple.elem1.loc5_38.2: ref %i32 = tuple_access %tuple.elem0.loc5_38.4, element1
// CHECK:STDOUT:   %.loc5_38.6: init %i32 = initialize_from %.loc5_38.5 to %tuple.elem1.loc5_38.2
// CHECK:STDOUT:   %.loc5_38.7: init %tuple.type.7c47 = tuple_init (%.loc5_38.4, %.loc5_38.6) to %tuple.elem0.loc5_38.4
// CHECK:STDOUT:   %.loc5_38.8: init %tuple.type.7c47 = converted %tuple.elem0.loc5_38.1, %.loc5_38.7
// CHECK:STDOUT:   %tuple.elem1.loc5_38.3: ref %tuple.type.e076 = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_38.6: ref %i32 = tuple_access %tuple.elem1.loc5_38.3, element0
// CHECK:STDOUT:   %.loc5_38.9: %i32 = bind_value %tuple.elem0.loc5_38.6
// CHECK:STDOUT:   %tuple.elem1.loc5_38.4: ref %tuple.type.e076 = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_38.7: ref %i32 = tuple_access %tuple.elem1.loc5_38.4, element0
// CHECK:STDOUT:   %.loc5_38.10: init %i32 = initialize_from %.loc5_38.9 to %tuple.elem0.loc5_38.7
// CHECK:STDOUT:   %tuple.elem1.loc5_38.5: ref %i32 = tuple_access %tuple.elem1.loc5_38.3, element1
// CHECK:STDOUT:   %.loc5_38.11: %i32 = bind_value %tuple.elem1.loc5_38.5
// CHECK:STDOUT:   %tuple.elem1.loc5_38.6: ref %i32 = tuple_access %tuple.elem1.loc5_38.4, element1
// CHECK:STDOUT:   %.loc5_38.12: init %i32 = initialize_from %.loc5_38.11 to %tuple.elem1.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.13: init %tuple.type.e076 = tuple_init (%.loc5_38.10, %.loc5_38.12) to %tuple.elem1.loc5_38.4
// CHECK:STDOUT:   %.loc5_38.14: init %tuple.type.e076 = converted %tuple.elem1.loc5_38.3, %.loc5_38.13
// CHECK:STDOUT:   %.loc5_38.15: init %tuple.type.fced = tuple_init (%.loc5_38.8, %.loc5_38.14) to file.%b.var
// CHECK:STDOUT:   %.loc5_43: init %tuple.type.fced = converted %b_ref.ref, %.loc5_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc5_43
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.0dce [template = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.fd9a = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.fd9a = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:   %X.patt => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.type.e076: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_2.ed97: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.e076 = tuple_value (%int_1.0593, %int_2.ed97) [template]
// CHECK:STDOUT:   %C.fd9a: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.b531 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.a359 = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.c9bd: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.0dce: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.2e04 = import_ref Implicit//default, inst1052 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.b531
// CHECK:STDOUT:     .b_ref = imports.%import_ref.a359
// CHECK:STDOUT:     .C = imports.%import_ref.c9bd
// CHECK:STDOUT:     .F = imports.%import_ref.0dce
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.e076) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2e04
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.fd9a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.0dce [template = constants.%F]
// CHECK:STDOUT:   %.loc12: ref %C.fd9a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.fd9a = call %F.ref() to %.loc12
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:   %X.patt => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.type.e076: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_3.781a: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_4.1ad1: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %tuple.96bf: %tuple.type.e076 = tuple_value (%int_3.781a, %int_4.1ad1) [template]
// CHECK:STDOUT:   %C.5864: type = class_type @C, @C(%tuple.96bf) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.b209: %tuple.type.e076 = tuple_value (%int_1, %int_2) [template]
// CHECK:STDOUT:   %C.fd9a: type = class_type @C, @C(%tuple.b209) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.b531 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.a359 = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.c9bd: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.0dce: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.edd8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.2e04 = import_ref Implicit//default, inst1052 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.b531
// CHECK:STDOUT:     .b_ref = imports.%import_ref.a359
// CHECK:STDOUT:     .C = imports.%import_ref.c9bd
// CHECK:STDOUT:     .F = imports.%import_ref.0dce
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %c_bad.var: ref %C.5864 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.5864 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.e076) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.e076 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.e076 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2e04
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.fd9a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.0dce [template = constants.%F]
// CHECK:STDOUT:   %.loc10_26: ref %C.fd9a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.fd9a = call %F.ref() to %.loc10_26
// CHECK:STDOUT:   %.loc10_27: %C.5864 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.96bf) {
// CHECK:STDOUT:   %X => constants.%tuple.96bf
// CHECK:STDOUT:   %X.patt => constants.%tuple.96bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.b209) {
// CHECK:STDOUT:   %X => constants.%tuple.b209
// CHECK:STDOUT:   %X.patt => constants.%tuple.b209
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
