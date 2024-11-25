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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Base, .d: %i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.14: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.15: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.16: type = struct_type {.base: %.14, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.40: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.13, %Convert.14 [template]
// CHECK:STDOUT:   %.42: <specific function> = specific_function %.41, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.43: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct.1: %Base = struct_value (%.43) [template]
// CHECK:STDOUT:   %.44: <bound method> = bound_method %.15, %Convert.14 [template]
// CHECK:STDOUT:   %.45: <specific function> = specific_function %.44, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.46: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %struct.2: %Derived = struct_value (%struct.1, %.46) [template]
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
// CHECK:STDOUT:     %.loc17_27: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc17_27: init type = call constants.%Int(%.loc17_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc17_32: init type = call constants.%Int(%.loc17_32) [template = constants.%i32]
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
// CHECK:STDOUT:   %.loc4_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc4_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_10.3: type = converted %int.make_type_signed, %.loc4_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8: %.2 = field_decl b, element0 [template]
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc4_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %.6 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %.loc10_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc10_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_10.3: type = converted %int.make_type_signed, %.loc10_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .d = %.loc10_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_25: Core.IntLiteral = int_value 4 [template = constants.%.13]
// CHECK:STDOUT:   %.loc14_26.1: %.14 = struct_literal (%.loc14_25)
// CHECK:STDOUT:   %.loc14_34: Core.IntLiteral = int_value 7 [template = constants.%.15]
// CHECK:STDOUT:   %.loc14_35.1: %.16 = struct_literal (%.loc14_26.1, %.loc14_34)
// CHECK:STDOUT:   %.loc14_26.2: %Convert.type.2 = interface_witness_access constants.%.40, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_26.3: <bound method> = bound_method %.loc14_25, %.loc14_26.2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc14_26.4: <specific function> = specific_function %.loc14_26.3, @Convert.2(constants.%.1) [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc14_26: init %i32 = call %.loc14_26.4(%.loc14_25) [template = constants.%.43]
// CHECK:STDOUT:   %.loc14_26.5: init %i32 = converted %.loc14_25, %int.convert_checked.loc14_26 [template = constants.%.43]
// CHECK:STDOUT:   %.loc14_35.2: ref %Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc14_26.6: ref %i32 = class_element_access %.loc14_35.2, element0
// CHECK:STDOUT:   %.loc14_26.7: init %i32 = initialize_from %.loc14_26.5 to %.loc14_26.6 [template = constants.%.43]
// CHECK:STDOUT:   %.loc14_26.8: init %Base = class_init (%.loc14_26.7), %.loc14_35.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc14_35.3: init %Base = converted %.loc14_26.1, %.loc14_26.8 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc14_35.4: %Convert.type.2 = interface_witness_access constants.%.40, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_35.5: <bound method> = bound_method %.loc14_34, %.loc14_35.4 [template = constants.%.44]
// CHECK:STDOUT:   %.loc14_35.6: <specific function> = specific_function %.loc14_35.5, @Convert.2(constants.%.1) [template = constants.%.45]
// CHECK:STDOUT:   %int.convert_checked.loc14_35: init %i32 = call %.loc14_35.6(%.loc14_34) [template = constants.%.46]
// CHECK:STDOUT:   %.loc14_35.7: init %i32 = converted %.loc14_34, %int.convert_checked.loc14_35 [template = constants.%.46]
// CHECK:STDOUT:   %.loc14_35.8: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc14_35.9: init %i32 = initialize_from %.loc14_35.7 to %.loc14_35.8 [template = constants.%.46]
// CHECK:STDOUT:   %.loc14_35.10: init %Derived = class_init (%.loc14_35.3, %.loc14_35.9), %return [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_36: init %Derived = converted %.loc14_35.1, %.loc14_35.10 [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc14_36 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param_patt: %Derived) -> %return: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc18_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc18_12: %.7 = name_ref d, @Derived.%.loc10_8 [template = @Derived.%.loc10_8]
// CHECK:STDOUT:   %.loc18_12.1: ref %i32 = class_element_access %d.ref.loc18_11, element1
// CHECK:STDOUT:   %.loc18_12.2: %i32 = bind_value %.loc18_12.1
// CHECK:STDOUT:   %d.ref.loc18_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %.6 = name_ref base, @Derived.%.loc8 [template = @Derived.%.loc8]
// CHECK:STDOUT:   %.loc18_17.1: ref %Base = class_element_access %d.ref.loc18_16, element0
// CHECK:STDOUT:   %.loc18_17.2: %Base = bind_value %.loc18_17.1
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, @Base.%.loc4_8 [template = @Base.%.loc4_8]
// CHECK:STDOUT:   %.loc18_22.1: ref %i32 = class_element_access %.loc18_17.2, element0
// CHECK:STDOUT:   %.loc18_22.2: %i32 = bind_value %.loc18_22.1
// CHECK:STDOUT:   %.loc18_24.1: %tuple.type.2 = tuple_literal (%.loc18_12.2, %.loc18_22.2)
// CHECK:STDOUT:   %.loc18_24.2: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_24.3: init %i32 = initialize_from %.loc18_12.2 to %.loc18_24.2
// CHECK:STDOUT:   %.loc18_24.4: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_24.5: init %i32 = initialize_from %.loc18_22.2 to %.loc18_24.4
// CHECK:STDOUT:   %.loc18_24.6: init %tuple.type.2 = tuple_init (%.loc18_24.3, %.loc18_24.5) to %return
// CHECK:STDOUT:   %.loc18_25: init %tuple.type.2 = converted %.loc18_24.1, %.loc18_24.6
// CHECK:STDOUT:   return %.loc18_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_after_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.d: %i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
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
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.loc7_10.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc7_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.3: type = converted %int.make_type_signed, %.loc7_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8: %.4 = field_decl d, element0 [template]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .d = %.loc7_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
