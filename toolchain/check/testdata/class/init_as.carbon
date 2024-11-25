// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_as.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() -> i32 {
  return ({.a = 1, .b = 2} as Class).a;
}

// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %Class = struct_value (%.35, %.38) [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc16_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_11.3: type = converted %int.make_type_signed, %.loc16_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.3: type = converted %int.make_type_signed.loc13, %.loc13_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_17: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_25: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_26.1: %.7 = struct_literal (%.loc17_17, %.loc17_25)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc17_26.2: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_26.3: <bound method> = bound_method %.loc17_17, %.loc17_26.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_26.4: <specific function> = specific_function %.loc17_26.3, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc17_26.1: init %i32 = call %.loc17_26.4(%.loc17_17) [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_26.5: init %i32 = converted %.loc17_17, %int.convert_checked.loc17_26.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_26.6: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc17_26.7: ref %i32 = class_element_access %.loc17_26.6, element0
// CHECK:STDOUT:   %.loc17_26.8: init %i32 = initialize_from %.loc17_26.5 to %.loc17_26.7 [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_26.9: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_26.10: <bound method> = bound_method %.loc17_25, %.loc17_26.9 [template = constants.%.36]
// CHECK:STDOUT:   %.loc17_26.11: <specific function> = specific_function %.loc17_26.10, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc17_26.2: init %i32 = call %.loc17_26.11(%.loc17_25) [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_26.12: init %i32 = converted %.loc17_25, %int.convert_checked.loc17_26.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_26.13: ref %i32 = class_element_access %.loc17_26.6, element1
// CHECK:STDOUT:   %.loc17_26.14: init %i32 = initialize_from %.loc17_26.12 to %.loc17_26.13 [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_26.15: init %Class = class_init (%.loc17_26.8, %.loc17_26.14), %.loc17_26.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_26.16: ref %Class = temporary %.loc17_26.6, %.loc17_26.15
// CHECK:STDOUT:   %.loc17_28: ref %Class = converted %.loc17_26.1, %.loc17_26.16
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Class.%.loc12_8 [template = @Class.%.loc12_8]
// CHECK:STDOUT:   %.loc17_37.1: ref %i32 = class_element_access %.loc17_28, element0
// CHECK:STDOUT:   %.loc17_37.2: %i32 = bind_value %.loc17_37.1
// CHECK:STDOUT:   return %.loc17_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
