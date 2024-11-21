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

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C(<cannot stringify inst+400 kind TupleValue>)` to `C(<cannot stringify inst+388 kind TupleValue>)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C(<cannot stringify inst+400 kind TupleValue>)` does not implement interface `ImplicitAs(C(<cannot stringify inst+388 kind TupleValue>))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%.29) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (%tuple.type.4, %tuple.type.5) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%tuple.type.2, %i32) [template]
// CHECK:STDOUT:   %tuple.type.8: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.9: type = tuple_type (%tuple.type.7, %tuple.type.8) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.11: type = tuple_type (%tuple.type.3, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.12: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.13: type = tuple_type (%tuple.type.11, %tuple.type.12) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.33, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.7 = tuple_value (%tuple.1, %.38) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.35, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple.3: %tuple.type.8 = tuple_value (%.41, %.44) [template]
// CHECK:STDOUT:   %tuple.4: %tuple.type.9 = tuple_value (%tuple.2, %tuple.3) [template]
// CHECK:STDOUT:   %X: %tuple.type.8 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.8 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %.45: type = struct_type {} [template]
// CHECK:STDOUT:   %.46: <witness> = complete_type_witness %.45 [template]
// CHECK:STDOUT:   %tuple.5: %tuple.type.8 = tuple_value (%.38, %.41) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple.5) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %.loc4_13: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc4)
// CHECK:STDOUT:   %.loc4_17.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.3: type = converted %int.make_type_signed.loc4, %.loc4_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.4: type = converted %.loc4_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.2 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %tuple.type.2 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc5_15: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_15: init type = call constants.%Int(%.loc5_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc5_15)
// CHECK:STDOUT:   %.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_22: init type = call constants.%Int(%.loc5_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_25: %tuple.type.4 = tuple_literal (%.loc5_19, %int.make_type_signed.loc5_22)
// CHECK:STDOUT:   %.loc5_29: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_29: init type = call constants.%Int(%.loc5_29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_34: init type = call constants.%Int(%.loc5_34) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_37: %tuple.type.5 = tuple_literal (%int.make_type_signed.loc5_29, %int.make_type_signed.loc5_34)
// CHECK:STDOUT:   %.loc5_38.1: %tuple.type.6 = tuple_literal (%.loc5_25, %.loc5_37)
// CHECK:STDOUT:   %.loc5_38.2: type = value_of_initializer %int.make_type_signed.loc5_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.3: type = converted %int.make_type_signed.loc5_15, %.loc5_38.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.4: type = converted %.loc5_19, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc5_38.5: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.6: type = converted %int.make_type_signed.loc5_22, %.loc5_38.5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.7: type = converted %.loc5_25, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %.loc5_38.8: type = value_of_initializer %int.make_type_signed.loc5_29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.9: type = converted %int.make_type_signed.loc5_29, %.loc5_38.8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.10: type = value_of_initializer %int.make_type_signed.loc5_34 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.11: type = converted %int.make_type_signed.loc5_34, %.loc5_38.10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_38.12: type = converted %.loc5_37, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:   %.loc5_38.13: type = converted %.loc5_38.1, constants.%tuple.type.9 [template = constants.%tuple.type.9]
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.9 = var b_ref
// CHECK:STDOUT:   %b_ref: ref %tuple.type.9 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %X.patt.loc7_9.1: %tuple.type.8 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: %tuple.type.8 = value_param_pattern %X.patt.loc7_9.1, runtime_param<invalid> [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_14: init type = call constants.%Int(%.loc7_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%.loc7_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.1: %tuple.type.5 = tuple_literal (%int.make_type_signed.loc7_14, %int.make_type_signed.loc7_19)
// CHECK:STDOUT:     %.loc7_22.2: type = value_of_initializer %int.make_type_signed.loc7_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.3: type = converted %int.make_type_signed.loc7_14, %.loc7_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.4: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.5: type = converted %int.make_type_signed.loc7_19, %.loc7_22.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.6: type = converted %.loc7_22.1, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:     %X.param: %tuple.type.8 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %X.loc7_9.1: %tuple.type.8 = bind_symbolic_name X, 0, %X.param [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc9_14: Core.IntLiteral = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:     %.loc9_17: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:     %.loc9_18.1: %tuple.type.12 = tuple_literal (%.loc9_14, %.loc9_17)
// CHECK:STDOUT:     %.loc9_18.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc9_18.3: <bound method> = bound_method %.loc9_14, %.loc9_18.2 [template = constants.%.36]
// CHECK:STDOUT:     %.loc9_18.4: <specific function> = specific_function %.loc9_18.3, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.1: init %i32 = call %.loc9_18.4(%.loc9_14) [template = constants.%.38]
// CHECK:STDOUT:     %.loc9_18.5: %i32 = value_of_initializer %int.convert_checked.loc9_18.1 [template = constants.%.38]
// CHECK:STDOUT:     %.loc9_18.6: %i32 = converted %.loc9_14, %.loc9_18.5 [template = constants.%.38]
// CHECK:STDOUT:     %.loc9_18.7: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc9_18.8: <bound method> = bound_method %.loc9_17, %.loc9_18.7 [template = constants.%.39]
// CHECK:STDOUT:     %.loc9_18.9: <specific function> = specific_function %.loc9_18.8, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.2: init %i32 = call %.loc9_18.9(%.loc9_17) [template = constants.%.41]
// CHECK:STDOUT:     %.loc9_18.10: %i32 = value_of_initializer %int.convert_checked.loc9_18.2 [template = constants.%.41]
// CHECK:STDOUT:     %.loc9_18.11: %i32 = converted %.loc9_17, %.loc9_18.10 [template = constants.%.41]
// CHECK:STDOUT:     %tuple: %tuple.type.8 = tuple_value (%.loc9_18.6, %.loc9_18.11) [template = constants.%tuple.5]
// CHECK:STDOUT:     %.loc9_12: %tuple.type.8 = converted %.loc9_18.1, %tuple [template = constants.%tuple.5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.5) [template = constants.%C.3]
// CHECK:STDOUT:     %return.param: ref %C.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.3 = return_slot %return.param
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
// CHECK:STDOUT:     %.loc7_26: <witness> = complete_type_witness %.45 [template = constants.%.46]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_22: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_24.1: %tuple.type.3 = tuple_literal (%.loc4_22)
// CHECK:STDOUT:   %.loc4_24.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_24.3: <bound method> = bound_method %.loc4_22, %.loc4_24.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_24.4: <specific function> = specific_function %.loc4_24.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %.loc4_24.4(%.loc4_22) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_24.5: init %i32 = converted %.loc4_22, %int.convert_checked.loc4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_24.6: init %tuple.type.2 = tuple_init (%.loc4_24.5) to file.%a_ref.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc4_25: init %tuple.type.2 = converted %.loc4_24.1, %.loc4_24.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_25
// CHECK:STDOUT:   %.loc5_45: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_47.1: %tuple.type.3 = tuple_literal (%.loc5_45)
// CHECK:STDOUT:   %.loc5_50: Core.IntLiteral = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc5_51.1: %tuple.type.11 = tuple_literal (%.loc5_47.1, %.loc5_50)
// CHECK:STDOUT:   %.loc5_55: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc5_58: Core.IntLiteral = int_value 3 [template = constants.%.35]
// CHECK:STDOUT:   %.loc5_59.1: %tuple.type.12 = tuple_literal (%.loc5_55, %.loc5_58)
// CHECK:STDOUT:   %.loc5_60.1: %tuple.type.13 = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %.loc5_47.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_47.3: <bound method> = bound_method %.loc5_45, %.loc5_47.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc5_47.4: <specific function> = specific_function %.loc5_47.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc5_47: init %i32 = call %.loc5_47.4(%.loc5_45) [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_47.5: init %i32 = converted %.loc5_45, %int.convert_checked.loc5_47 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_60.2: ref %tuple.type.7 = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc5_51.2: ref %tuple.type.2 = tuple_access %.loc5_60.2, element0
// CHECK:STDOUT:   %.loc5_47.6: init %tuple.type.2 = tuple_init (%.loc5_47.5) to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.3: init %tuple.type.2 = converted %.loc5_47.1, %.loc5_47.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.4: init %tuple.type.2 = initialize_from %.loc5_51.3 to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.5: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_51.6: <bound method> = bound_method %.loc5_50, %.loc5_51.5 [template = constants.%.36]
// CHECK:STDOUT:   %.loc5_51.7: <specific function> = specific_function %.loc5_51.6, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc5_51: init %i32 = call %.loc5_51.7(%.loc5_50) [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_51.8: init %i32 = converted %.loc5_50, %int.convert_checked.loc5_51 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_51.9: ref %i32 = tuple_access %.loc5_60.2, element1
// CHECK:STDOUT:   %.loc5_51.10: init %i32 = initialize_from %.loc5_51.8 to %.loc5_51.9 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_51.11: init %tuple.type.7 = tuple_init (%.loc5_51.4, %.loc5_51.10) to %.loc5_60.2 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.3: init %tuple.type.7 = converted %.loc5_51.1, %.loc5_51.11 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_59.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_59.3: <bound method> = bound_method %.loc5_55, %.loc5_59.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_59.4: <specific function> = specific_function %.loc5_59.3, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.1: init %i32 = call %.loc5_59.4(%.loc5_55) [template = constants.%.41]
// CHECK:STDOUT:   %.loc5_59.5: init %i32 = converted %.loc5_55, %int.convert_checked.loc5_59.1 [template = constants.%.41]
// CHECK:STDOUT:   %.loc5_60.4: ref %tuple.type.8 = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc5_59.6: ref %i32 = tuple_access %.loc5_60.4, element0
// CHECK:STDOUT:   %.loc5_59.7: init %i32 = initialize_from %.loc5_59.5 to %.loc5_59.6 [template = constants.%.41]
// CHECK:STDOUT:   %.loc5_59.8: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_59.9: <bound method> = bound_method %.loc5_58, %.loc5_59.8 [template = constants.%.42]
// CHECK:STDOUT:   %.loc5_59.10: <specific function> = specific_function %.loc5_59.9, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.2: init %i32 = call %.loc5_59.10(%.loc5_58) [template = constants.%.44]
// CHECK:STDOUT:   %.loc5_59.11: init %i32 = converted %.loc5_58, %int.convert_checked.loc5_59.2 [template = constants.%.44]
// CHECK:STDOUT:   %.loc5_59.12: ref %i32 = tuple_access %.loc5_60.4, element1
// CHECK:STDOUT:   %.loc5_59.13: init %i32 = initialize_from %.loc5_59.11 to %.loc5_59.12 [template = constants.%.44]
// CHECK:STDOUT:   %.loc5_59.14: init %tuple.type.8 = tuple_init (%.loc5_59.7, %.loc5_59.13) to %.loc5_60.4 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.5: init %tuple.type.8 = converted %.loc5_59.1, %.loc5_59.14 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.6: init %tuple.type.9 = tuple_init (%.loc5_60.3, %.loc5_60.5) to file.%b_ref.var [template = constants.%tuple.4]
// CHECK:STDOUT:   %.loc5_61: init %tuple.type.9 = converted %.loc5_60.1, %.loc5_60.6 [template = constants.%tuple.4]
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
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Convert.2) [symbolic]
// CHECK:STDOUT:   %.5: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.3, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.4, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.5 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.18 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%iN)> [symbolic]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = facet_access_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %As.type.2, %Convert.type.7 [symbolic]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.28 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.7, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %.15: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.8, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.9, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.5, %Convert.type.11 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.21 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%tuple.type.3, %tuple.type.4) [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (%tuple.type.2, %i32) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.8: type = tuple_type (%tuple.type.6, %tuple.type.7) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %X: %tuple.type.7 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.7 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.10: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.29: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.30: %.29 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.31: %.2 = assoc_entity element0, imports.%import_ref.42 [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.21) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.27, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.21) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.28, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.21) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.7 = tuple_value (%.35, %.38) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %tuple.type.2 = import_ref Implicit//default, inst+26, loaded
// CHECK:STDOUT:   %import_ref.2: ref %tuple.type.8 = import_ref Implicit//default, inst+412, loaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+495, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+529, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.39
// CHECK:STDOUT:     .ImplicitAs = %import_ref.41
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.6: @ImplicitAs.%.1 (%.2) = import_ref Implicit//default, inst+45, loaded [symbolic = @ImplicitAs.%.2 (constants.%.31)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+91, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+92, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Implicit//default, inst+93, loaded [symbolic = @impl.1.%.1 (constants.%.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+62, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+121, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+122, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+123, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+150, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+151, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+152, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+182, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+183, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+184, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+215, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+216, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+222, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+223, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+224, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+240, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+279, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+280, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+281, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+308, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+309, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+310, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+340, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+341, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+342, unloaded
// CHECK:STDOUT:   %import_ref.40 = import_ref Implicit//default, inst+500, unloaded
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
// CHECK:STDOUT:   %.loc4_9: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc4)
// CHECK:STDOUT:   %.loc4_13.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.3: type = converted %int.make_type_signed.loc4, %.loc4_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.4: type = converted %.loc4_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc5_11: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_11: init type = call constants.%Int(%.loc5_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_15: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc5_11)
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_18: init type = call constants.%Int(%.loc5_18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21: %tuple.type.3 = tuple_literal (%.loc5_15, %int.make_type_signed.loc5_18)
// CHECK:STDOUT:   %.loc5_25: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_25: init type = call constants.%Int(%.loc5_25) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_30: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_30: init type = call constants.%Int(%.loc5_30) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_33: %tuple.type.4 = tuple_literal (%int.make_type_signed.loc5_25, %int.make_type_signed.loc5_30)
// CHECK:STDOUT:   %.loc5_34.1: %tuple.type.5 = tuple_literal (%.loc5_21, %.loc5_33)
// CHECK:STDOUT:   %.loc5_34.2: type = value_of_initializer %int.make_type_signed.loc5_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.3: type = converted %int.make_type_signed.loc5_11, %.loc5_34.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.4: type = converted %.loc5_15, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc5_34.5: type = value_of_initializer %int.make_type_signed.loc5_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.6: type = converted %int.make_type_signed.loc5_18, %.loc5_34.5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.7: type = converted %.loc5_21, constants.%tuple.type.6 [template = constants.%tuple.type.6]
// CHECK:STDOUT:   %.loc5_34.8: type = value_of_initializer %int.make_type_signed.loc5_25 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.9: type = converted %int.make_type_signed.loc5_25, %.loc5_34.8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.10: type = value_of_initializer %int.make_type_signed.loc5_30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.11: type = converted %int.make_type_signed.loc5_30, %.loc5_34.10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_34.12: type = converted %.loc5_33, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %.loc5_34.13: type = converted %.loc5_34.1, constants.%tuple.type.8 [template = constants.%tuple.type.8]
// CHECK:STDOUT:   %b.var: ref %tuple.type.8 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.8 = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc6_11: Core.IntLiteral = int_value 1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc6_14: Core.IntLiteral = int_value 2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc6_15.1: %tuple.type.10 = tuple_literal (%.loc6_11, %.loc6_14)
// CHECK:STDOUT:   %.loc6_15.2: %Convert.type.13 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_15.3: <bound method> = bound_method %.loc6_11, %.loc6_15.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc6_15.4: <specific function> = specific_function %.loc6_15.3, @Convert.2(constants.%.21) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc6_15.1: init %i32 = call %.loc6_15.4(%.loc6_11) [template = constants.%.35]
// CHECK:STDOUT:   %.loc6_15.5: %i32 = value_of_initializer %int.convert_checked.loc6_15.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc6_15.6: %i32 = converted %.loc6_11, %.loc6_15.5 [template = constants.%.35]
// CHECK:STDOUT:   %.loc6_15.7: %Convert.type.13 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_15.8: <bound method> = bound_method %.loc6_14, %.loc6_15.7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc6_15.9: <specific function> = specific_function %.loc6_15.8, @Convert.2(constants.%.21) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc6_15.2: init %i32 = call %.loc6_15.9(%.loc6_14) [template = constants.%.38]
// CHECK:STDOUT:   %.loc6_15.10: %i32 = value_of_initializer %int.convert_checked.loc6_15.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc6_15.11: %i32 = converted %.loc6_14, %.loc6_15.10 [template = constants.%.38]
// CHECK:STDOUT:   %tuple: %tuple.type.7 = tuple_value (%.loc6_15.6, %.loc6_15.11) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_9: %tuple.type.7 = converted %.loc6_15.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple) [template = constants.%C.3]
// CHECK:STDOUT:   %c.var: ref %C.3 = var c
// CHECK:STDOUT:   %c: ref %C.3 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .Convert = imports.%import_ref.6
// CHECK:STDOUT:     witness = (imports.%import_ref.7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.7) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.12) = assoc_entity element0, imports.%import_ref.28 [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.22
// CHECK:STDOUT:     .Convert = imports.%import_ref.23
// CHECK:STDOUT:     witness = (imports.%import_ref.24)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.1(%N) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @impl.1.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%iN)> [symbolic = %As.type (constants.%As.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%uN)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.36 as imports.%import_ref.37 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.7) {
// CHECK:STDOUT:   %X: %tuple.type.7 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.7 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.1)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%.1 (%.11)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.2 = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   %.loc4_17.1: ref %i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_17.2: %i32 = bind_value %.loc4_17.1
// CHECK:STDOUT:   %.loc4_17.3: init %tuple.type.2 = tuple_init (%.loc4_17.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_22: init %tuple.type.2 = converted %a_ref.ref, %.loc4_17.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_22
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.8 = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   %.loc5_38.1: ref %tuple.type.6 = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_38.2: ref %tuple.type.2 = tuple_access %.loc5_38.1, element0
// CHECK:STDOUT:   %.loc5_38.3: ref %i32 = tuple_access %.loc5_38.2, element0
// CHECK:STDOUT:   %.loc5_38.4: %i32 = bind_value %.loc5_38.3
// CHECK:STDOUT:   %.loc5_38.5: ref %tuple.type.6 = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_38.6: ref %tuple.type.2 = tuple_access %.loc5_38.5, element0
// CHECK:STDOUT:   %.loc5_38.7: init %tuple.type.2 = tuple_init (%.loc5_38.4) to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.8: init %tuple.type.2 = converted %.loc5_38.2, %.loc5_38.7
// CHECK:STDOUT:   %.loc5_38.9: init %tuple.type.2 = initialize_from %.loc5_38.8 to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.10: ref %i32 = tuple_access %.loc5_38.1, element1
// CHECK:STDOUT:   %.loc5_38.11: %i32 = bind_value %.loc5_38.10
// CHECK:STDOUT:   %.loc5_38.12: ref %i32 = tuple_access %.loc5_38.5, element1
// CHECK:STDOUT:   %.loc5_38.13: init %i32 = initialize_from %.loc5_38.11 to %.loc5_38.12
// CHECK:STDOUT:   %.loc5_38.14: init %tuple.type.6 = tuple_init (%.loc5_38.9, %.loc5_38.13) to %.loc5_38.5
// CHECK:STDOUT:   %.loc5_38.15: init %tuple.type.6 = converted %.loc5_38.1, %.loc5_38.14
// CHECK:STDOUT:   %.loc5_38.16: ref %tuple.type.7 = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_38.17: ref %i32 = tuple_access %.loc5_38.16, element0
// CHECK:STDOUT:   %.loc5_38.18: %i32 = bind_value %.loc5_38.17
// CHECK:STDOUT:   %.loc5_38.19: ref %tuple.type.7 = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_38.20: ref %i32 = tuple_access %.loc5_38.19, element0
// CHECK:STDOUT:   %.loc5_38.21: init %i32 = initialize_from %.loc5_38.18 to %.loc5_38.20
// CHECK:STDOUT:   %.loc5_38.22: ref %i32 = tuple_access %.loc5_38.16, element1
// CHECK:STDOUT:   %.loc5_38.23: %i32 = bind_value %.loc5_38.22
// CHECK:STDOUT:   %.loc5_38.24: ref %i32 = tuple_access %.loc5_38.19, element1
// CHECK:STDOUT:   %.loc5_38.25: init %i32 = initialize_from %.loc5_38.23 to %.loc5_38.24
// CHECK:STDOUT:   %.loc5_38.26: init %tuple.type.7 = tuple_init (%.loc5_38.21, %.loc5_38.25) to %.loc5_38.19
// CHECK:STDOUT:   %.loc5_38.27: init %tuple.type.7 = converted %.loc5_38.16, %.loc5_38.26
// CHECK:STDOUT:   %.loc5_38.28: init %tuple.type.8 = tuple_init (%.loc5_38.15, %.loc5_38.27) to file.%b.var
// CHECK:STDOUT:   %.loc5_43: init %tuple.type.8 = converted %b_ref.ref, %.loc5_38.28
// CHECK:STDOUT:   assign file.%b.var, %.loc5_43
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.3 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.1.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.5.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.6.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.11
// CHECK:STDOUT:   %Convert => constants.%Convert.11
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %Dest => constants.%i32
// CHECK:STDOUT:   %Dest.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.13
// CHECK:STDOUT:   %.1 => constants.%.29
// CHECK:STDOUT:   %.2 => constants.%.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%.21) {
// CHECK:STDOUT:   %N => constants.%.21
// CHECK:STDOUT:   %N.patt => constants.%.21
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.14
// CHECK:STDOUT:   %Convert => constants.%Convert.14
// CHECK:STDOUT:   %.1 => constants.%.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%.21) {
// CHECK:STDOUT:   %N => constants.%.21
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Convert.2) [symbolic]
// CHECK:STDOUT:   %.5: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.3, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.4, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.5 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.18 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%iN)> [symbolic]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = facet_access_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %As.type.2, %Convert.type.7 [symbolic]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.28 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.7, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %.15: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.8, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.9, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.5, %Convert.type.11 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.23: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.23 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %.24: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.26: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.28: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.29, %.28) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+412, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+495, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+529, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Implicit//default, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+91, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+92, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Implicit//default, inst+93, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+62, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+121, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+122, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+123, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+150, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+151, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+152, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+182, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+183, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+184, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+215, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+216, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+222, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+223, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+224, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+240, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+279, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+280, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+281, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+308, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+309, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+310, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+340, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+341, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+342, unloaded
// CHECK:STDOUT:   %import_ref.39 = import_ref Implicit//default, inst+500, unloaded
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
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc12_15: Core.IntLiteral = int_value 1 [template = constants.%.24]
// CHECK:STDOUT:   %.loc12_18: Core.IntLiteral = int_value 2 [template = constants.%.25]
// CHECK:STDOUT:   %.loc12_21: Core.IntLiteral = int_value 3 [template = constants.%.26]
// CHECK:STDOUT:   %.loc12_22: %tuple.type.2 = tuple_literal (%.loc12_15, %.loc12_18, %.loc12_21)
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .Convert = imports.%import_ref.6
// CHECK:STDOUT:     witness = (imports.%import_ref.7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.7) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.12) = assoc_entity element0, imports.%import_ref.28 [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.22
// CHECK:STDOUT:     .Convert = imports.%import_ref.23
// CHECK:STDOUT:     witness = (imports.%import_ref.24)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.1(%N) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @impl.1.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%iN)> [symbolic = %As.type (constants.%As.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%uN)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.36 as imports.%import_ref.37 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.1) {
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.39
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.1)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%.1 (%.11)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc12: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc12
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.1.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.5.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.6.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.11
// CHECK:STDOUT:   %Convert => constants.%Convert.11
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Convert.2) [symbolic]
// CHECK:STDOUT:   %.5: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.3, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.4, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.5 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.18 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%iN)> [symbolic]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = facet_access_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %As.type.2, %Convert.type.7 [symbolic]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.28 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.7, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %.15: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.8, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.9, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.5, %Convert.type.11 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.23: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.23 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %.24: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.27: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.28: %.27 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.29: %.2 = assoc_entity element0, imports.%import_ref.41 [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.23) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.24, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.23) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.25, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.23) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.1 = tuple_value (%.33, %.36) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple.1) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.1 = tuple_value (%.39, %.38) [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%tuple.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.7: type = facet_type <@ImplicitAs, @ImplicitAs(%C.3)> [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.1, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.40: type = assoc_entity_type %ImplicitAs.type.7, %Convert.type.15 [template]
// CHECK:STDOUT:   %.41: %.40 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+412, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+495, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+529, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.40
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.6: @ImplicitAs.%.1 (%.2) = import_ref Implicit//default, inst+45, loaded [symbolic = @ImplicitAs.%.2 (constants.%.29)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+91, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+92, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Implicit//default, inst+93, loaded [symbolic = @impl.1.%.1 (constants.%.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+62, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+121, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+122, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+123, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+150, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+151, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+152, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+182, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+183, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+184, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+215, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+216, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+222, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+223, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+224, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+240, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+279, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+280, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+281, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+308, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+309, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+310, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+340, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+341, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+342, unloaded
// CHECK:STDOUT:   %import_ref.39 = import_ref Implicit//default, inst+500, unloaded
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
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc10_15: Core.IntLiteral = int_value 3 [template = constants.%.24]
// CHECK:STDOUT:   %.loc10_18: Core.IntLiteral = int_value 4 [template = constants.%.25]
// CHECK:STDOUT:   %.loc10_19.1: %tuple.type.2 = tuple_literal (%.loc10_15, %.loc10_18)
// CHECK:STDOUT:   %.loc10_19.2: %Convert.type.13 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_19.3: <bound method> = bound_method %.loc10_15, %.loc10_19.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc10_19.4: <specific function> = specific_function %.loc10_19.3, @Convert.2(constants.%.23) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc10_19.1: init %i32 = call %.loc10_19.4(%.loc10_15) [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_19.5: %i32 = value_of_initializer %int.convert_checked.loc10_19.1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_19.6: %i32 = converted %.loc10_15, %.loc10_19.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_19.7: %Convert.type.13 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_19.8: <bound method> = bound_method %.loc10_18, %.loc10_19.7 [template = constants.%.34]
// CHECK:STDOUT:   %.loc10_19.9: <specific function> = specific_function %.loc10_19.8, @Convert.2(constants.%.23) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc10_19.2: init %i32 = call %.loc10_19.9(%.loc10_18) [template = constants.%.36]
// CHECK:STDOUT:   %.loc10_19.10: %i32 = value_of_initializer %int.convert_checked.loc10_19.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc10_19.11: %i32 = converted %.loc10_18, %.loc10_19.10 [template = constants.%.36]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc10_19.6, %.loc10_19.11) [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc10_13: %tuple.type.1 = converted %.loc10_19.1, %tuple [template = constants.%tuple.1]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%tuple.1) [template = constants.%C.3]
// CHECK:STDOUT:   %c_bad.var: ref %C.3 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.3 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .Convert = imports.%import_ref.6
// CHECK:STDOUT:     witness = (imports.%import_ref.7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.7) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.12) = assoc_entity element0, imports.%import_ref.28 [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.22
// CHECK:STDOUT:     .Convert = imports.%import_ref.23
// CHECK:STDOUT:     witness = (imports.%import_ref.24)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.1(%N) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @impl.1.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%iN)> [symbolic = %As.type (constants.%As.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%uN)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.32 as imports.%import_ref.33 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.36 as imports.%import_ref.37 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%X: %tuple.type.1) {
// CHECK:STDOUT:   %X: %tuple.type.1 = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.39
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.1)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%.1 (%.11)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.4;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc10_25: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.4 = call %F.ref() to %.loc10_25
// CHECK:STDOUT:   %.loc10_27: %C.3 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.1.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.5.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.6.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.11
// CHECK:STDOUT:   %Convert => constants.%Convert.11
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %Dest => constants.%i32
// CHECK:STDOUT:   %Dest.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.13
// CHECK:STDOUT:   %.1 => constants.%.27
// CHECK:STDOUT:   %.2 => constants.%.28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%.23) {
// CHECK:STDOUT:   %N => constants.%.23
// CHECK:STDOUT:   %N.patt => constants.%.23
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.14
// CHECK:STDOUT:   %Convert => constants.%Convert.14
// CHECK:STDOUT:   %.1 => constants.%.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%.23) {
// CHECK:STDOUT:   %N => constants.%.23
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.1) {
// CHECK:STDOUT:   %X => constants.%tuple.1
// CHECK:STDOUT:   %X.patt => constants.%tuple.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.2) {
// CHECK:STDOUT:   %X => constants.%tuple.2
// CHECK:STDOUT:   %X.patt => constants.%tuple.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C.3) {
// CHECK:STDOUT:   %Dest => constants.%C.3
// CHECK:STDOUT:   %Dest.patt => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.15
// CHECK:STDOUT:   %Convert => constants.%Convert.15
// CHECK:STDOUT:   %.1 => constants.%.40
// CHECK:STDOUT:   %.2 => constants.%.41
// CHECK:STDOUT: }
// CHECK:STDOUT:
