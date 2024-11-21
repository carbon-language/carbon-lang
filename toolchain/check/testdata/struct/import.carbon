// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

class C(S:! {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});

// --- implicit.impl.carbon

impl package Implicit;

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C(<cannot stringify inst+400 kind StructValue>)` to `C(<cannot stringify inst+388 kind StructValue>)` [ImplicitAsConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C(<cannot stringify inst+400 kind StructValue>)` does not implement interface `ImplicitAs(C(<cannot stringify inst+388 kind StructValue>))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C({.a = 3, .b = 4}) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct.1: %.2 = struct_value (%.31) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.32: type = struct_type {.b: %i32, .c: %tuple.type.2} [template]
// CHECK:STDOUT:   %.33: type = struct_type {.a: %.32, .d: %i32} [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.37: type = struct_type {.b: Core.IntLiteral, .c: %tuple.type.3} [template]
// CHECK:STDOUT:   %.38: type = struct_type {.a: %.37, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.31) [template]
// CHECK:STDOUT:   %struct.2: %.32 = struct_value (%.31, %tuple) [template]
// CHECK:STDOUT:   %struct.3: %.33 = struct_value (%struct.2, %.31) [template]
// CHECK:STDOUT:   %.39: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %.39 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %.39 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %.40: type = struct_type {} [template]
// CHECK:STDOUT:   %.41: <witness> = complete_type_witness %.40 [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.43: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.44: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.42, %Convert.14 [template]
// CHECK:STDOUT:   %.47: <specific function> = specific_function %.46, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.48: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.49: <bound method> = bound_method %.43, %Convert.14 [template]
// CHECK:STDOUT:   %.50: <specific function> = specific_function %.49, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.51: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct.4: %.39 = struct_value (%.48, %.51) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct.4) [template]
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
// CHECK:STDOUT:   %.loc4_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_17.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17.3: type = converted %int.make_type_signed.loc4, %.loc4_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_20: type = struct_type {.a: %i32} [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref %.2 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %.2 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc5_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_22: init type = call constants.%Int(%.loc5_22.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_22.2: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_22.3: type = converted %int.make_type_signed.loc5_22, %.loc5_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_32: init type = call constants.%Int(%.loc5_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_36.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc5_32)
// CHECK:STDOUT:   %.loc5_36.2: type = value_of_initializer %int.make_type_signed.loc5_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_36.3: type = converted %int.make_type_signed.loc5_32, %.loc5_36.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_36.4: type = converted %.loc5_36.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc5_37: type = struct_type {.b: %i32, .c: %tuple.type.2} [template = constants.%.32]
// CHECK:STDOUT:   %.loc5_44.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_44: init type = call constants.%Int(%.loc5_44.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_44.2: type = value_of_initializer %int.make_type_signed.loc5_44 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_44.3: type = converted %int.make_type_signed.loc5_44, %.loc5_44.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_47: type = struct_type {.a: %.32, .d: %i32} [template = constants.%.33]
// CHECK:STDOUT:   %b_ref.var: ref %.33 = var b_ref
// CHECK:STDOUT:   %b_ref: ref %.33 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %S.patt.loc8_9.1: %.39 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:     %S.param_patt: %.39 = value_param_pattern %S.patt.loc8_9.1, runtime_param<invalid> [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc8_18: init type = call constants.%Int(%.loc8_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_18.2: type = value_of_initializer %int.make_type_signed.loc8_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_18.3: type = converted %int.make_type_signed.loc8_18, %.loc8_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc8_27: init type = call constants.%Int(%.loc8_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.2: type = value_of_initializer %int.make_type_signed.loc8_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.3: type = converted %int.make_type_signed.loc8_27, %.loc8_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_30: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.39]
// CHECK:STDOUT:     %S.param: %.39 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %S.loc8_9.1: %.39 = bind_symbolic_name S, 0, %S.param [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc9_19: Core.IntLiteral = int_value 1 [template = constants.%.42]
// CHECK:STDOUT:     %.loc9_27: Core.IntLiteral = int_value 2 [template = constants.%.43]
// CHECK:STDOUT:     %.loc9_28.1: %.44 = struct_literal (%.loc9_19, %.loc9_27)
// CHECK:STDOUT:     %.loc9_28.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc9_28.3: <bound method> = bound_method %.loc9_19, %.loc9_28.2 [template = constants.%.46]
// CHECK:STDOUT:     %.loc9_28.4: <specific function> = specific_function %.loc9_28.3, @Convert.2(constants.%.1) [template = constants.%.47]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.1: init %i32 = call %.loc9_28.4(%.loc9_19) [template = constants.%.48]
// CHECK:STDOUT:     %.loc9_28.5: %i32 = value_of_initializer %int.convert_checked.loc9_28.1 [template = constants.%.48]
// CHECK:STDOUT:     %.loc9_28.6: %i32 = converted %.loc9_19, %.loc9_28.5 [template = constants.%.48]
// CHECK:STDOUT:     %.loc9_28.7: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc9_28.8: <bound method> = bound_method %.loc9_27, %.loc9_28.7 [template = constants.%.49]
// CHECK:STDOUT:     %.loc9_28.9: <specific function> = specific_function %.loc9_28.8, @Convert.2(constants.%.1) [template = constants.%.50]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.2: init %i32 = call %.loc9_28.9(%.loc9_27) [template = constants.%.51]
// CHECK:STDOUT:     %.loc9_28.10: %i32 = value_of_initializer %int.convert_checked.loc9_28.2 [template = constants.%.51]
// CHECK:STDOUT:     %.loc9_28.11: %i32 = converted %.loc9_27, %.loc9_28.10 [template = constants.%.51]
// CHECK:STDOUT:     %struct: %.39 = struct_value (%.loc9_28.6, %.loc9_28.11) [template = constants.%struct.4]
// CHECK:STDOUT:     %.loc9_12: %.39 = converted %.loc9_28.1, %struct [template = constants.%struct.4]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.4) [template = constants.%C.3]
// CHECK:STDOUT:     %return.param: ref %C.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc8_9.1: %.39) {
// CHECK:STDOUT:   %S.loc8_9.2: %.39 = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc8_9.2: %.39 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc8_34: <witness> = complete_type_witness %.40 [template = constants.%.41]
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
// CHECK:STDOUT:   %.loc4_30: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_31.1: %.4 = struct_literal (%.loc4_30)
// CHECK:STDOUT:   %.loc4_31.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_31.3: <bound method> = bound_method %.loc4_30, %.loc4_31.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_31.4: <specific function> = specific_function %.loc4_31.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %.loc4_31.4(%.loc4_30) [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_31.5: init %i32 = converted %.loc4_30, %int.convert_checked.loc4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_31.6: init %.2 = struct_init (%.loc4_31.5) to file.%a_ref.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc4_32: init %.2 = converted %.loc4_31.1, %.loc4_31.6 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_32
// CHECK:STDOUT:   %.loc6_17: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_26: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_28.1: %tuple.type.3 = tuple_literal (%.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: %.37 = struct_literal (%.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %.loc6_37: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_38.1: %.38 = struct_literal (%.loc6_29.1, %.loc6_37)
// CHECK:STDOUT:   %.loc6_29.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_29.3: <bound method> = bound_method %.loc6_17, %.loc6_29.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc6_29.4: <specific function> = specific_function %.loc6_29.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc6_29: init %i32 = call %.loc6_29.4(%.loc6_17) [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_29.5: init %i32 = converted %.loc6_17, %int.convert_checked.loc6_29 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_38.2: ref %.32 = struct_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc6_29.6: ref %i32 = struct_access %.loc6_38.2, element0
// CHECK:STDOUT:   %.loc6_29.7: init %i32 = initialize_from %.loc6_29.5 to %.loc6_29.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_28.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_28.3: <bound method> = bound_method %.loc6_26, %.loc6_28.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc6_28.4: <specific function> = specific_function %.loc6_28.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc6_28: init %i32 = call %.loc6_28.4(%.loc6_26) [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_28.5: init %i32 = converted %.loc6_26, %int.convert_checked.loc6_28 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_29.8: ref %tuple.type.2 = struct_access %.loc6_38.2, element1
// CHECK:STDOUT:   %.loc6_28.6: init %tuple.type.2 = tuple_init (%.loc6_28.5) to %.loc6_29.8 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.9: init %tuple.type.2 = converted %.loc6_28.1, %.loc6_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.10: init %tuple.type.2 = initialize_from %.loc6_29.9 to %.loc6_29.8 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.11: init %.32 = struct_init (%.loc6_29.7, %.loc6_29.10) to %.loc6_38.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc6_38.3: init %.32 = converted %.loc6_29.1, %.loc6_29.11 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc6_38.4: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_38.5: <bound method> = bound_method %.loc6_37, %.loc6_38.4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc6_38.6: <specific function> = specific_function %.loc6_38.5, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc6_38: init %i32 = call %.loc6_38.6(%.loc6_37) [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_38.7: init %i32 = converted %.loc6_37, %int.convert_checked.loc6_38 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_38.8: ref %i32 = struct_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc6_38.9: init %i32 = initialize_from %.loc6_38.7 to %.loc6_38.8 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_38.10: init %.33 = struct_init (%.loc6_38.3, %.loc6_38.9) to file.%b_ref.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc6_39: init %.33 = converted %.loc6_38.1, %.loc6_38.10 [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc6_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.4) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%struct.4
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%struct.4
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
// CHECK:STDOUT:   %.22: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.23: type = struct_type {.b: %i32, .c: %tuple.type.2} [template]
// CHECK:STDOUT:   %.24: type = struct_type {.a: %.23, .d: %i32} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.30: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %.30 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %.30 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.33: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.35: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.36: %.35 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.37: %.2 = assoc_entity element0, imports.%import_ref.42 [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.21) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.38: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.21) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.32, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.21) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.30 = struct_value (%.41, %.44) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %.22 = import_ref Implicit//default, inst+24, loaded
// CHECK:STDOUT:   %import_ref.2: ref %.24 = import_ref Implicit//default, inst+399, loaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+458, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+502, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.39
// CHECK:STDOUT:     .ImplicitAs = %import_ref.41
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6: @ImplicitAs.%.1 (%.2) = import_ref Implicit//default, inst+43, loaded [symbolic = @ImplicitAs.%.2 (constants.%.37)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+89, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+90, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Implicit//default, inst+91, loaded [symbolic = @impl.1.%.1 (constants.%.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+60, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+119, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+120, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+121, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+148, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+149, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+150, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+180, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+181, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+182, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+212, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+213, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+220, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+221, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+222, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+238, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+277, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+278, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+279, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+306, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+307, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+308, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+338, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+339, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+340, unloaded
// CHECK:STDOUT:   %import_ref.40 = import_ref Implicit//default, inst+463, unloaded
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
// CHECK:STDOUT:   %.loc4_13.1: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.3: type = converted %int.make_type_signed.loc4, %.loc4_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_16: type = struct_type {.a: %i32} [template = constants.%.22]
// CHECK:STDOUT:   %a.var: ref %.22 = var a
// CHECK:STDOUT:   %a: ref %.22 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc5_18.1: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_18: init type = call constants.%Int(%.loc5_18.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.2: type = value_of_initializer %int.make_type_signed.loc5_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.3: type = converted %int.make_type_signed.loc5_18, %.loc5_18.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_28: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_28: init type = call constants.%Int(%.loc5_28) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc5_28)
// CHECK:STDOUT:   %.loc5_32.2: type = value_of_initializer %int.make_type_signed.loc5_28 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32.3: type = converted %int.make_type_signed.loc5_28, %.loc5_32.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32.4: type = converted %.loc5_32.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc5_33: type = struct_type {.b: %i32, .c: %tuple.type.2} [template = constants.%.23]
// CHECK:STDOUT:   %.loc5_40.1: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed.loc5_40: init type = call constants.%Int(%.loc5_40.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_40.2: type = value_of_initializer %int.make_type_signed.loc5_40 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_40.3: type = converted %int.make_type_signed.loc5_40, %.loc5_40.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_43: type = struct_type {.a: %.23, .d: %i32} [template = constants.%.24]
// CHECK:STDOUT:   %b.var: ref %.24 = var b
// CHECK:STDOUT:   %b: ref %.24 = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc6_16: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_24: Core.IntLiteral = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc6_25.1: %.33 = struct_literal (%.loc6_16, %.loc6_24)
// CHECK:STDOUT:   %.loc6_25.2: %Convert.type.13 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_25.3: <bound method> = bound_method %.loc6_16, %.loc6_25.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc6_25.4: <specific function> = specific_function %.loc6_25.3, @Convert.2(constants.%.21) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc6_25.1: init %i32 = call %.loc6_25.4(%.loc6_16) [template = constants.%.41]
// CHECK:STDOUT:   %.loc6_25.5: %i32 = value_of_initializer %int.convert_checked.loc6_25.1 [template = constants.%.41]
// CHECK:STDOUT:   %.loc6_25.6: %i32 = converted %.loc6_16, %.loc6_25.5 [template = constants.%.41]
// CHECK:STDOUT:   %.loc6_25.7: %Convert.type.13 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_25.8: <bound method> = bound_method %.loc6_24, %.loc6_25.7 [template = constants.%.42]
// CHECK:STDOUT:   %.loc6_25.9: <specific function> = specific_function %.loc6_25.8, @Convert.2(constants.%.21) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc6_25.2: init %i32 = call %.loc6_25.9(%.loc6_24) [template = constants.%.44]
// CHECK:STDOUT:   %.loc6_25.10: %i32 = value_of_initializer %int.convert_checked.loc6_25.2 [template = constants.%.44]
// CHECK:STDOUT:   %.loc6_25.11: %i32 = converted %.loc6_24, %.loc6_25.10 [template = constants.%.44]
// CHECK:STDOUT:   %struct: %.30 = struct_value (%.loc6_25.6, %.loc6_25.11) [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_9: %.30 = converted %.loc6_25.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%struct) [template = constants.%C.3]
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
// CHECK:STDOUT: generic class @C(constants.%S: %.30) {
// CHECK:STDOUT:   %S: %.30 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %.30 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
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
// CHECK:STDOUT:   %a_ref.ref: ref %.22 = name_ref a_ref, imports.%import_ref.1
// CHECK:STDOUT:   %.loc4_20.1: ref %i32 = struct_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_20.2: %i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init %.22 = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_25: init %.22 = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_25
// CHECK:STDOUT:   %b_ref.ref: ref %.24 = name_ref b_ref, imports.%import_ref.2
// CHECK:STDOUT:   %.loc5_47.1: ref %.23 = struct_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_47.2: ref %i32 = struct_access %.loc5_47.1, element0
// CHECK:STDOUT:   %.loc5_47.3: %i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref %.23 = struct_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_47.5: ref %i32 = struct_access %.loc5_47.4, element0
// CHECK:STDOUT:   %.loc5_47.6: init %i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref %tuple.type.2 = struct_access %.loc5_47.1, element1
// CHECK:STDOUT:   %.loc5_47.8: ref %i32 = tuple_access %.loc5_47.7, element0
// CHECK:STDOUT:   %.loc5_47.9: %i32 = bind_value %.loc5_47.8
// CHECK:STDOUT:   %.loc5_47.10: ref %tuple.type.2 = struct_access %.loc5_47.4, element1
// CHECK:STDOUT:   %.loc5_47.11: init %tuple.type.2 = tuple_init (%.loc5_47.9) to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init %tuple.type.2 = converted %.loc5_47.7, %.loc5_47.11
// CHECK:STDOUT:   %.loc5_47.13: init %tuple.type.2 = initialize_from %.loc5_47.12 to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.14: init %.23 = struct_init (%.loc5_47.6, %.loc5_47.13) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.15: init %.23 = converted %.loc5_47.1, %.loc5_47.14
// CHECK:STDOUT:   %.loc5_47.16: ref %i32 = struct_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_47.17: %i32 = bind_value %.loc5_47.16
// CHECK:STDOUT:   %.loc5_47.18: ref %i32 = struct_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_47.19: init %i32 = initialize_from %.loc5_47.17 to %.loc5_47.18
// CHECK:STDOUT:   %.loc5_47.20: init %.24 = struct_init (%.loc5_47.15, %.loc5_47.19) to file.%b.var
// CHECK:STDOUT:   %.loc5_52: init %.24 = converted %b_ref.ref, %.loc5_47.20
// CHECK:STDOUT:   assign file.%b.var, %.loc5_52
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
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
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
// CHECK:STDOUT:   %.1 => constants.%.35
// CHECK:STDOUT:   %.2 => constants.%.36
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
// CHECK:STDOUT:   %.1 => constants.%.38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%.21) {
// CHECK:STDOUT:   %N => constants.%.21
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%struct
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
// CHECK:STDOUT:   %.24: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %.24 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %.24 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.26: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.27: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct: %.24 = struct_value (%.30, %.29) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+399, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+458, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+502, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Implicit//default, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+89, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+90, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Implicit//default, inst+91, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+60, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+119, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+120, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+121, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+148, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+149, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+150, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+180, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+181, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+182, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+212, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+213, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+220, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+221, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+222, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+238, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+277, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+278, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+279, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+306, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+307, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+308, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+338, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+339, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+340, unloaded
// CHECK:STDOUT:   %import_ref.39 = import_ref Implicit//default, inst+463, unloaded
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
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%.25]
// CHECK:STDOUT:   %.loc11_28: Core.IntLiteral = int_value 2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc11_29: %.27 = struct_literal (%.loc11_20, %.loc11_28)
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
// CHECK:STDOUT: generic class @C(constants.%S: %.24) {
// CHECK:STDOUT:   %S: %.24 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %.24 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
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
// CHECK:STDOUT:   %.loc11: ref %C.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.3 = call %F.ref() to %.loc11
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
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%struct
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
// CHECK:STDOUT:   %.24: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %.24 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %.24 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.26: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.27: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.29: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.30: %.29 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.31: %.2 = assoc_entity element0, imports.%import_ref.41 [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.23) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.25, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.23) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.26, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.23) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct.1: %.24 = struct_value (%.35, %.38) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%struct.1) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.2: %.24 = struct_value (%.41, %.40) [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%struct.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.7: type = facet_type <@ImplicitAs, @ImplicitAs(%C.3)> [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.1, @ImplicitAs(%C.3) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.42: type = assoc_entity_type %ImplicitAs.type.7, %Convert.type.15 [template]
// CHECK:STDOUT:   %.43: %.42 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Implicit//default, inst+399, unloaded
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Implicit//default, inst+458, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Implicit//default, inst+502, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.40
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Implicit//default, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.6: @ImplicitAs.%.1 (%.2) = import_ref Implicit//default, inst+43, loaded [symbolic = @ImplicitAs.%.2 (constants.%.31)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Implicit//default, inst+89, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Implicit//default, inst+90, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Implicit//default, inst+91, loaded [symbolic = @impl.1.%.1 (constants.%.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Implicit//default, inst+60, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Implicit//default, inst+119, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Implicit//default, inst+120, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Implicit//default, inst+121, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Implicit//default, inst+148, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Implicit//default, inst+149, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.17 = import_ref Implicit//default, inst+150, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Implicit//default, inst+180, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.20: type = import_ref Implicit//default, inst+181, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.21 = import_ref Implicit//default, inst+182, unloaded
// CHECK:STDOUT:   %import_ref.22 = import_ref Implicit//default, inst+212, unloaded
// CHECK:STDOUT:   %import_ref.23 = import_ref Implicit//default, inst+213, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Implicit//default, inst+214, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Implicit//default, inst+220, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Implicit//default, inst+221, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.27 = import_ref Implicit//default, inst+222, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Implicit//default, inst+238, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Implicit//default, inst+277, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Implicit//default, inst+278, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.31 = import_ref Implicit//default, inst+279, unloaded
// CHECK:STDOUT:   %import_ref.32: type = import_ref Implicit//default, inst+306, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.33: type = import_ref Implicit//default, inst+307, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.34 = import_ref Implicit//default, inst+308, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Implicit//default, inst+338, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Implicit//default, inst+339, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.38 = import_ref Implicit//default, inst+340, unloaded
// CHECK:STDOUT:   %import_ref.39 = import_ref Implicit//default, inst+463, unloaded
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
// CHECK:STDOUT:   %.loc9_20: Core.IntLiteral = int_value 3 [template = constants.%.25]
// CHECK:STDOUT:   %.loc9_28: Core.IntLiteral = int_value 4 [template = constants.%.26]
// CHECK:STDOUT:   %.loc9_29.1: %.27 = struct_literal (%.loc9_20, %.loc9_28)
// CHECK:STDOUT:   %.loc9_29.2: %Convert.type.13 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_29.3: <bound method> = bound_method %.loc9_20, %.loc9_29.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc9_29.4: <specific function> = specific_function %.loc9_29.3, @Convert.2(constants.%.23) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc9_29.1: init %i32 = call %.loc9_29.4(%.loc9_20) [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_29.5: %i32 = value_of_initializer %int.convert_checked.loc9_29.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_29.6: %i32 = converted %.loc9_20, %.loc9_29.5 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_29.7: %Convert.type.13 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_29.8: <bound method> = bound_method %.loc9_28, %.loc9_29.7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_29.9: <specific function> = specific_function %.loc9_29.8, @Convert.2(constants.%.23) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc9_29.2: init %i32 = call %.loc9_29.9(%.loc9_28) [template = constants.%.38]
// CHECK:STDOUT:   %.loc9_29.10: %i32 = value_of_initializer %int.convert_checked.loc9_29.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc9_29.11: %i32 = converted %.loc9_28, %.loc9_29.10 [template = constants.%.38]
// CHECK:STDOUT:   %struct: %.24 = struct_value (%.loc9_29.6, %.loc9_29.11) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc9_13: %.24 = converted %.loc9_29.1, %struct [template = constants.%struct.1]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%struct.1) [template = constants.%C.3]
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
// CHECK:STDOUT: generic class @C(constants.%S: %.24) {
// CHECK:STDOUT:   %S: %.24 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %.24 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
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
// CHECK:STDOUT:   %.loc9_35: ref %C.4 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.4 = call %F.ref() to %.loc9_35
// CHECK:STDOUT:   %.loc9_37: %C.3 = converted %F.call, <error> [template = <error>]
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
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
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
// CHECK:STDOUT: specific @impl.1(constants.%.23) {
// CHECK:STDOUT:   %N => constants.%.23
// CHECK:STDOUT:   %N.patt => constants.%.23
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.14
// CHECK:STDOUT:   %Convert => constants.%Convert.14
// CHECK:STDOUT:   %.1 => constants.%.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%.23) {
// CHECK:STDOUT:   %N => constants.%.23
// CHECK:STDOUT:   %iN => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.1) {
// CHECK:STDOUT:   %S => constants.%struct.1
// CHECK:STDOUT:   %S.patt => constants.%struct.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.2) {
// CHECK:STDOUT:   %S => constants.%struct.2
// CHECK:STDOUT:   %S.patt => constants.%struct.2
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
// CHECK:STDOUT:   %.1 => constants.%.42
// CHECK:STDOUT:   %.2 => constants.%.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
