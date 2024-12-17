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

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+6]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
// CHECK:STDERR:
impl package Implicit;

var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;
var c: C((1, 2)) = F();

// --- fail_bad_type.impl.carbon

// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+6]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+3]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
// CHECK:STDERR:
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

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+5]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+2]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon: error: semantics TODO: `TryResolveInst on Call` [SemanticsTodo]
impl package Implicit;

var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (%tuple.type.4, %tuple.type.5) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%tuple.type.2, %i32) [template]
// CHECK:STDOUT:   %tuple.type.8: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.9: type = tuple_type (%tuple.type.7, %tuple.type.8) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.11: type = tuple_type (%tuple.type.3, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.12: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.13: type = tuple_type (%tuple.type.11, %tuple.type.12) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_1) [symbolic]
// CHECK:STDOUT:   %tuple.2: %tuple.type.7 = tuple_value (%tuple.1, %int.convert_checked.2) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %Int = call %Convert.specific_fn.3(%int_2) [symbolic]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %Int = call %Convert.specific_fn.4(%int_3) [symbolic]
// CHECK:STDOUT:   %tuple.3: %tuple.type.8 = tuple_value (%int.convert_checked.3, %int.convert_checked.4) [symbolic]
// CHECK:STDOUT:   %tuple.4: %tuple.type.9 = tuple_value (%tuple.2, %tuple.3) [symbolic]
// CHECK:STDOUT:   %X: %tuple.type.8 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.8 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.5: %tuple.type.8 = tuple_value (%int.convert_checked.2, %int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%tuple.5) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:   %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.1: %tuple.type.1 = tuple_literal (%i32.loc4)
// CHECK:STDOUT:   %.loc4_17.2: type = converted %.loc4_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.2 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %tuple.type.2 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %int_32.loc5_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: %tuple.type.1 = tuple_literal (%i32.loc5_15)
// CHECK:STDOUT:   %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_25: %tuple.type.4 = tuple_literal (%.loc5_19, %i32.loc5_22)
// CHECK:STDOUT:   %int_32.loc5_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_37: %tuple.type.5 = tuple_literal (%i32.loc5_29, %i32.loc5_34)
// CHECK:STDOUT:   %.loc5_38.1: %tuple.type.6 = tuple_literal (%.loc5_25, %.loc5_37)
// CHECK:STDOUT:   %.loc5_38.2: type = converted %.loc5_19, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc5_38.3: type = converted %.loc5_25, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %.loc5_38.4: type = converted %.loc5_37, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:   %.loc5_38.5: type = converted %.loc5_38.1, constants.%tuple.type.9 [template = constants.%tuple.type.9]
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.9 = var b_ref
// CHECK:STDOUT:   %b_ref: ref %tuple.type.9 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %X.patt.loc7_9.1: %tuple.type.8 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: %tuple.type.8 = value_param_pattern %X.patt.loc7_9.1, runtime_param<invalid> [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.1: %tuple.type.5 = tuple_literal (%i32.loc7_14, %i32.loc7_19)
// CHECK:STDOUT:     %.loc7_22.2: type = converted %.loc7_22.1, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:     %X.param: %tuple.type.8 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %X.loc7_9.1: %tuple.type.8 = bind_symbolic_name X, 0, %X.param [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:     %.loc9_18.1: %tuple.type.12 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_18.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound.loc9_18.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_18.1 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_18.1: <specific function> = specific_function %Convert.bound.loc9_18.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.1: init %Int = call %Convert.specific_fn.loc9_18.1(%int_1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:     %.loc9_18.2: %i32 = value_of_initializer %int.convert_checked.loc9_18.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:     %.loc9_18.3: %i32 = converted %int_1, %.loc9_18.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:     %impl.elem0.loc9_18.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound.loc9_18.2: <bound method> = bound_method %int_2, %impl.elem0.loc9_18.2 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_18.2: <specific function> = specific_function %Convert.bound.loc9_18.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.2: init %Int = call %Convert.specific_fn.loc9_18.2(%int_2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:     %.loc9_18.4: %i32 = value_of_initializer %int.convert_checked.loc9_18.2 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:     %.loc9_18.5: %i32 = converted %int_2, %.loc9_18.4 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:     %tuple: %tuple.type.8 = tuple_value (%.loc9_18.3, %.loc9_18.5) [symbolic = constants.%tuple.5]
// CHECK:STDOUT:     %.loc9_19: %tuple.type.8 = converted %.loc9_18.1, %tuple [symbolic = constants.%tuple.5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.5) [symbolic = constants.%C.2]
// CHECK:STDOUT:     %return.param: ref %C.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc7_9.1: %tuple.type.8) {
// CHECK:STDOUT:   %X.loc7_9.2: %tuple.type.8 = bind_symbolic_name X, 0 [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc7_9.2: %tuple.type.8 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.4]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc4_24.1: %tuple.type.3 = tuple_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %Int = call %Convert.specific_fn.loc4(%int_0.loc4) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc4_24.2: init %i32 = converted %int_0.loc4, %int.convert_checked.loc4 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc4_24.3: init %tuple.type.2 = tuple_init (%.loc4_24.2) to file.%a_ref.var [symbolic = constants.%tuple.1]
// CHECK:STDOUT:   %.loc4_25: init %tuple.type.2 = converted %.loc4_24.1, %.loc4_24.3 [symbolic = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_25
// CHECK:STDOUT:   %int_0.loc5: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc5_47.1: %tuple.type.3 = tuple_literal (%int_0.loc5)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc5_51.1: %tuple.type.11 = tuple_literal (%.loc5_47.1, %int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc5_59.1: %tuple.type.12 = tuple_literal (%int_2, %int_3)
// CHECK:STDOUT:   %.loc5_60.1: %tuple.type.13 = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %impl.elem0.loc5_47: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5_47: <bound method> = bound_method %int_0.loc5, %impl.elem0.loc5_47 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_47: <specific function> = specific_function %Convert.bound.loc5_47, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_47: init %Int = call %Convert.specific_fn.loc5_47(%int_0.loc5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_47.2: init %i32 = converted %int_0.loc5, %int.convert_checked.loc5_47 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %tuple.elem0.loc5_60: ref %tuple.type.7 = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_51: ref %tuple.type.2 = tuple_access %tuple.elem0.loc5_60, element0
// CHECK:STDOUT:   %.loc5_47.3: init %tuple.type.2 = tuple_init (%.loc5_47.2) to %tuple.elem0.loc5_51 [symbolic = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.2: init %tuple.type.2 = converted %.loc5_47.1, %.loc5_47.3 [symbolic = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.3: init %tuple.type.2 = initialize_from %.loc5_51.2 to %tuple.elem0.loc5_51 [symbolic = constants.%tuple.1]
// CHECK:STDOUT:   %impl.elem0.loc5_51: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5_51: <bound method> = bound_method %int_1, %impl.elem0.loc5_51 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_51: <specific function> = specific_function %Convert.bound.loc5_51, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc5_51: init %Int = call %Convert.specific_fn.loc5_51(%int_1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_51.4: init %i32 = converted %int_1, %int.convert_checked.loc5_51 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %tuple.elem1.loc5_51: ref %i32 = tuple_access %tuple.elem0.loc5_60, element1
// CHECK:STDOUT:   %.loc5_51.5: init %i32 = initialize_from %.loc5_51.4 to %tuple.elem1.loc5_51 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_51.6: init %tuple.type.7 = tuple_init (%.loc5_51.3, %.loc5_51.5) to %tuple.elem0.loc5_60 [symbolic = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.2: init %tuple.type.7 = converted %.loc5_51.1, %.loc5_51.6 [symbolic = constants.%tuple.2]
// CHECK:STDOUT:   %impl.elem0.loc5_59.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5_59.1: <bound method> = bound_method %int_2, %impl.elem0.loc5_59.1 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_59.1: <specific function> = specific_function %Convert.bound.loc5_59.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.1: init %Int = call %Convert.specific_fn.loc5_59.1(%int_2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc5_59.2: init %i32 = converted %int_2, %int.convert_checked.loc5_59.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %tuple.elem1.loc5_60: ref %tuple.type.8 = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element0
// CHECK:STDOUT:   %.loc5_59.3: init %i32 = initialize_from %.loc5_59.2 to %tuple.elem0.loc5_59 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %impl.elem0.loc5_59.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5_59.2: <bound method> = bound_method %int_3, %impl.elem0.loc5_59.2 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc5_59.2: <specific function> = specific_function %Convert.bound.loc5_59.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.2: init %Int = call %Convert.specific_fn.loc5_59.2(%int_3) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc5_59.4: init %i32 = converted %int_3, %int.convert_checked.loc5_59.2 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %tuple.elem1.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element1
// CHECK:STDOUT:   %.loc5_59.5: init %i32 = initialize_from %.loc5_59.4 to %tuple.elem1.loc5_59 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc5_59.6: init %tuple.type.8 = tuple_init (%.loc5_59.3, %.loc5_59.5) to %tuple.elem1.loc5_60 [symbolic = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.3: init %tuple.type.8 = converted %.loc5_59.1, %.loc5_59.6 [symbolic = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.4: init %tuple.type.9 = tuple_init (%.loc5_60.2, %.loc5_60.3) to file.%b_ref.var [symbolic = constants.%tuple.4]
// CHECK:STDOUT:   %.loc5_61: init %tuple.type.9 = converted %.loc5_60.1, %.loc5_60.4 [symbolic = constants.%tuple.4]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5_61
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.5) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%tuple.5
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%tuple.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%tuple.type.3, %tuple.type.4) [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (%tuple.type.2, %i32) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.8: type = tuple_type (%tuple.type.6, %tuple.type.7) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %X: %tuple.type.7 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.7 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.10: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.2), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.2 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.2 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type.7 = tuple_value (%int.convert_checked.1, %int.convert_checked.2) [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%tuple) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %tuple.type.2 = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %import_ref.2: ref %tuple.type.8 = import_ref Implicit//default, b_ref, loaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.194
// CHECK:STDOUT:     .ImplicitAs = %import_ref.197
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.195: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.196 = import_ref Implicit//default, inst638 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_13.1: %tuple.type.1 = tuple_literal (%i32.loc10)
// CHECK:STDOUT:   %.loc10_13.2: type = converted %.loc10_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: %tuple.type.1 = tuple_literal (%i32.loc11_11)
// CHECK:STDOUT:   %int_32.loc11_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_21: %tuple.type.3 = tuple_literal (%.loc11_15, %i32.loc11_18)
// CHECK:STDOUT:   %int_32.loc11_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_30: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_33: %tuple.type.4 = tuple_literal (%i32.loc11_25, %i32.loc11_30)
// CHECK:STDOUT:   %.loc11_34.1: %tuple.type.5 = tuple_literal (%.loc11_21, %.loc11_33)
// CHECK:STDOUT:   %.loc11_34.2: type = converted %.loc11_15, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc11_34.3: type = converted %.loc11_21, constants.%tuple.type.6 [template = constants.%tuple.type.6]
// CHECK:STDOUT:   %.loc11_34.4: type = converted %.loc11_33, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %.loc11_34.5: type = converted %.loc11_34.1, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:   %b.var: ref %tuple.type.8 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.8 = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc12_15.1: %tuple.type.10 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc12_15.1: %Convert.type.9 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc12_15.1: <bound method> = bound_method %int_1, %impl.elem0.loc12_15.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_15.1: <specific function> = specific_function %Convert.bound.loc12_15.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_15.1: init %Int = call %Convert.specific_fn.loc12_15.1(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc12_15.2: %i32 = value_of_initializer %int.convert_checked.loc12_15.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc12_15.3: %i32 = converted %int_1, %.loc12_15.2 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc12_15.2: %Convert.type.9 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc12_15.2: <bound method> = bound_method %int_2, %impl.elem0.loc12_15.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_15.2: <specific function> = specific_function %Convert.bound.loc12_15.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_15.2: init %Int = call %Convert.specific_fn.loc12_15.2(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_15.4: %i32 = value_of_initializer %int.convert_checked.loc12_15.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_15.5: %i32 = converted %int_2, %.loc12_15.4 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %tuple: %tuple.type.7 = tuple_value (%.loc12_15.3, %.loc12_15.5) [symbolic = constants.%tuple]
// CHECK:STDOUT:   %.loc12_16: %tuple.type.7 = converted %.loc12_15.1, %tuple [symbolic = constants.%tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple) [symbolic = constants.%C.2]
// CHECK:STDOUT:   %c.var: ref %C.2 = var c
// CHECK:STDOUT:   %c: ref %C.2 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.7) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.7 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.7 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.196
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.195
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error> [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.2 = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   %tuple.elem0.loc10: ref %i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc10_17.1: %i32 = bind_value %tuple.elem0.loc10
// CHECK:STDOUT:   %.loc10_17.2: init %tuple.type.2 = tuple_init (%.loc10_17.1) to file.%a.var
// CHECK:STDOUT:   %.loc10_22: init %tuple.type.2 = converted %a_ref.ref, %.loc10_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc10_22
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.8 = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   %tuple.elem0.loc11_38.1: ref %tuple.type.6 = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_38.2: ref %tuple.type.2 = tuple_access %tuple.elem0.loc11_38.1, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_38.3: ref %i32 = tuple_access %tuple.elem0.loc11_38.2, element0
// CHECK:STDOUT:   %.loc11_38.1: %i32 = bind_value %tuple.elem0.loc11_38.3
// CHECK:STDOUT:   %tuple.elem0.loc11_38.4: ref %tuple.type.6 = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_38.5: ref %tuple.type.2 = tuple_access %tuple.elem0.loc11_38.4, element0
// CHECK:STDOUT:   %.loc11_38.2: init %tuple.type.2 = tuple_init (%.loc11_38.1) to %tuple.elem0.loc11_38.5
// CHECK:STDOUT:   %.loc11_38.3: init %tuple.type.2 = converted %tuple.elem0.loc11_38.2, %.loc11_38.2
// CHECK:STDOUT:   %.loc11_38.4: init %tuple.type.2 = initialize_from %.loc11_38.3 to %tuple.elem0.loc11_38.5
// CHECK:STDOUT:   %tuple.elem1.loc11_38.1: ref %i32 = tuple_access %tuple.elem0.loc11_38.1, element1
// CHECK:STDOUT:   %.loc11_38.5: %i32 = bind_value %tuple.elem1.loc11_38.1
// CHECK:STDOUT:   %tuple.elem1.loc11_38.2: ref %i32 = tuple_access %tuple.elem0.loc11_38.4, element1
// CHECK:STDOUT:   %.loc11_38.6: init %i32 = initialize_from %.loc11_38.5 to %tuple.elem1.loc11_38.2
// CHECK:STDOUT:   %.loc11_38.7: init %tuple.type.6 = tuple_init (%.loc11_38.4, %.loc11_38.6) to %tuple.elem0.loc11_38.4
// CHECK:STDOUT:   %.loc11_38.8: init %tuple.type.6 = converted %tuple.elem0.loc11_38.1, %.loc11_38.7
// CHECK:STDOUT:   %tuple.elem1.loc11_38.3: ref %tuple.type.7 = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %tuple.elem0.loc11_38.6: ref %i32 = tuple_access %tuple.elem1.loc11_38.3, element0
// CHECK:STDOUT:   %.loc11_38.9: %i32 = bind_value %tuple.elem0.loc11_38.6
// CHECK:STDOUT:   %tuple.elem1.loc11_38.4: ref %tuple.type.7 = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc11_38.7: ref %i32 = tuple_access %tuple.elem1.loc11_38.4, element0
// CHECK:STDOUT:   %.loc11_38.10: init %i32 = initialize_from %.loc11_38.9 to %tuple.elem0.loc11_38.7
// CHECK:STDOUT:   %tuple.elem1.loc11_38.5: ref %i32 = tuple_access %tuple.elem1.loc11_38.3, element1
// CHECK:STDOUT:   %.loc11_38.11: %i32 = bind_value %tuple.elem1.loc11_38.5
// CHECK:STDOUT:   %tuple.elem1.loc11_38.6: ref %i32 = tuple_access %tuple.elem1.loc11_38.4, element1
// CHECK:STDOUT:   %.loc11_38.12: init %i32 = initialize_from %.loc11_38.11 to %tuple.elem1.loc11_38.6
// CHECK:STDOUT:   %.loc11_38.13: init %tuple.type.7 = tuple_init (%.loc11_38.10, %.loc11_38.12) to %tuple.elem1.loc11_38.4
// CHECK:STDOUT:   %.loc11_38.14: init %tuple.type.7 = converted %tuple.elem1.loc11_38.3, %.loc11_38.13
// CHECK:STDOUT:   %.loc11_38.15: init %tuple.type.8 = tuple_init (%.loc11_38.8, %.loc11_38.14) to file.%b.var
// CHECK:STDOUT:   %.loc11_43: init %tuple.type.8 = converted %b_ref.ref, %.loc11_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc11_43
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref()
// CHECK:STDOUT:   assign file.%c.var, <error>
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
// CHECK:STDOUT: specific @C(<error>) {
// CHECK:STDOUT:   %X => <error>
// CHECK:STDOUT:   %X.patt => <error>
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
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.194: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.195 = import_ref Implicit//default, inst638 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc18: %tuple.type.2 = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.1) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.195
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.194
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error> [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref()
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(<error>) {
// CHECK:STDOUT:   %X => <error>
// CHECK:STDOUT:   %X.patt => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.2), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_3, %Convert.2 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_3) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_4, %Convert.2 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_4) [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%int.convert_checked.1, %int.convert_checked.2) [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%tuple) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.196
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.194: <witness> = import_ref Implicit//default, loc7_26, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.195 = import_ref Implicit//default, inst638 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.1
// CHECK:STDOUT:     .b_ref = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .F = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc9_19.1: %tuple.type.2 = tuple_literal (%int_3, %int_4)
// CHECK:STDOUT:   %impl.elem0.loc9_19.1: %Convert.type.9 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc9_19.1: <bound method> = bound_method %int_3, %impl.elem0.loc9_19.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_19.1: <specific function> = specific_function %Convert.bound.loc9_19.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_19.1: init %Int = call %Convert.specific_fn.loc9_19.1(%int_3) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_19.2: %i32 = value_of_initializer %int.convert_checked.loc9_19.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_19.3: %i32 = converted %int_3, %.loc9_19.2 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_19.2: %Convert.type.9 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.2]
// CHECK:STDOUT:   %Convert.bound.loc9_19.2: <bound method> = bound_method %int_4, %impl.elem0.loc9_19.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_19.2: <specific function> = specific_function %Convert.bound.loc9_19.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_19.2: init %Int = call %Convert.specific_fn.loc9_19.2(%int_4) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_19.4: %i32 = value_of_initializer %int.convert_checked.loc9_19.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_19.5: %i32 = converted %int_4, %.loc9_19.4 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc9_19.3, %.loc9_19.5) [symbolic = constants.%tuple]
// CHECK:STDOUT:   %.loc9_20: %tuple.type.1 = converted %.loc9_19.1, %tuple [symbolic = constants.%tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple) [symbolic = constants.%C.2]
// CHECK:STDOUT:   %c_bad.var: ref %C.2 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.2 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.1) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.195
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.194
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error> [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref()
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
// CHECK:STDOUT: specific @C(<error>) {
// CHECK:STDOUT:   %X => <error>
// CHECK:STDOUT:   %X.patt => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
