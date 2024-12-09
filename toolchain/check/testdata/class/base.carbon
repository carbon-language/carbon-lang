// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base.carbon

// --- base.carbon
package Base;

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

// --- fail_base_after_field.carbon
package BaseAfterField;

base class Base {
}

class Derived {
  var d: i32;

  // CHECK:STDERR: fail_base_after_field.carbon:[[@LINE+3]]:3: error: `base` declaration must appear before field declarations [BaseDeclAfterFieldDecl]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  extend base: Base;
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %struct_type.b.1: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.b.1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem.1: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %Derived.elem.2: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.d.1: type = struct_type {.base: %Base, .d: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.d.1 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.b.2: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %int_7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %struct_type.base.d.3: type = struct_type {.base: %struct_type.b.2, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_4.2) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_7.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val, %int_7.2) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Derived = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Derived = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %return.param: ref %Derived = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Derived = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %int_32.loc17_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc17_27: init type = call constants.%Int(%int_32.loc17_27) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc17_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc17_32: init type = call constants.%Int(%int_32.loc17_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc17_27, %int.make_type_signed.loc17_32)
// CHECK:STDOUT:     %.loc17_35.2: type = value_of_initializer %int.make_type_signed.loc17_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.3: type = converted %int.make_type_signed.loc17_27, %.loc17_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.4: type = value_of_initializer %int.make_type_signed.loc17_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.5: type = converted %int.make_type_signed.loc17_32, %.loc17_35.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.6: type = converted %.loc17_35.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %d.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_10.2: type = converted %int.make_type_signed, %.loc4_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8: %Base.elem = field_decl b, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b.1 [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc4_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem.1 = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_10.2: type = converted %int.make_type_signed, %.loc10_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8: %Derived.elem.2 = field_decl d, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .d = %.loc10_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %.loc14_26.1: %struct_type.b.2 = struct_literal (%int_4)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.1]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.base.d.3 = struct_literal (%.loc14_26.1, %int_7)
// CHECK:STDOUT:   %impl.elem0.loc14_26: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_26: <bound method> = bound_method %int_4, %impl.elem0.loc14_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_26: <specific function> = specific_function %Convert.bound.loc14_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_26: init %i32 = call %Convert.specific_fn.loc14_26(%int_4) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc14_26.2: init %i32 = converted %int_4, %int.convert_checked.loc14_26 [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc14_35.2: ref %Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc14_26.3: ref %i32 = class_element_access %.loc14_35.2, element0
// CHECK:STDOUT:   %.loc14_26.4: init %i32 = initialize_from %.loc14_26.2 to %.loc14_26.3 [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc14_26.5: init %Base = class_init (%.loc14_26.4), %.loc14_35.2 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc14_35.3: init %Base = converted %.loc14_26.1, %.loc14_26.5 [template = constants.%Base.val]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_35: <bound method> = bound_method %int_7, %impl.elem0.loc14_35 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_35: <specific function> = specific_function %Convert.bound.loc14_35, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_35: init %i32 = call %Convert.specific_fn.loc14_35(%int_7) [template = constants.%int_7.2]
// CHECK:STDOUT:   %.loc14_35.4: init %i32 = converted %int_7, %int.convert_checked.loc14_35 [template = constants.%int_7.2]
// CHECK:STDOUT:   %.loc14_35.5: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc14_35.6: init %i32 = initialize_from %.loc14_35.4 to %.loc14_35.5 [template = constants.%int_7.2]
// CHECK:STDOUT:   %.loc14_35.7: init %Derived = class_init (%.loc14_35.3, %.loc14_35.6), %return [template = constants.%Derived.val]
// CHECK:STDOUT:   %.loc14_36: init %Derived = converted %.loc14_35.1, %.loc14_35.7 [template = constants.%Derived.val]
// CHECK:STDOUT:   return %.loc14_36 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param_patt: %Derived) -> %return.param_patt: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc18_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc18_12: %Derived.elem.2 = name_ref d, @Derived.%.loc10_8 [template = @Derived.%.loc10_8]
// CHECK:STDOUT:   %.loc18_12.1: ref %i32 = class_element_access %d.ref.loc18_11, element1
// CHECK:STDOUT:   %.loc18_12.2: %i32 = bind_value %.loc18_12.1
// CHECK:STDOUT:   %d.ref.loc18_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %Derived.elem.1 = name_ref base, @Derived.%.loc8 [template = @Derived.%.loc8]
// CHECK:STDOUT:   %.loc18_17.1: ref %Base = class_element_access %d.ref.loc18_16, element0
// CHECK:STDOUT:   %.loc18_17.2: %Base = bind_value %.loc18_17.1
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc4_8 [template = @Base.%.loc4_8]
// CHECK:STDOUT:   %.loc18_22.1: ref %i32 = class_element_access %.loc18_17.2, element0
// CHECK:STDOUT:   %.loc18_22.2: %i32 = bind_value %.loc18_22.1
// CHECK:STDOUT:   %.loc18_24.1: %tuple.type.2 = tuple_literal (%.loc18_12.2, %.loc18_22.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_24.2: init %i32 = initialize_from %.loc18_12.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_24.3: init %i32 = initialize_from %.loc18_22.2 to %tuple.elem1
// CHECK:STDOUT:   %.loc18_24.4: init %tuple.type.2 = tuple_init (%.loc18_24.2, %.loc18_24.3) to %return
// CHECK:STDOUT:   %.loc18_25: init %tuple.type.2 = converted %.loc18_24.1, %.loc18_24.4
// CHECK:STDOUT:   return %.loc18_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_after_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.d [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.2: type = converted %int.make_type_signed, %.loc7_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8: %Derived.elem = field_decl d, element0 [template]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.d [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .d = %.loc7_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
