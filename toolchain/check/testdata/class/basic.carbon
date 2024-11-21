// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/basic.carbon

class Class {
  fn F(n: i32) -> i32 {
    return n;
  }

  fn G(n: i32) -> i32;

  var k: i32;
}

fn Class.G(n: i32) -> i32 {
  return n;
}

fn Run() -> i32 {
  return Class.F(4);
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 4 [template]
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
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc21_15: init type = call constants.%Int(%.loc21_15.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_15.2: type = value_of_initializer %int.make_type_signed.loc21_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_15.3: type = converted %int.make_type_signed.loc21_15, %.loc21_15.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc21_23: init type = call constants.%Int(%.loc21_23.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_23.2: type = value_of_initializer %int.make_type_signed.loc21_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_23.3: type = converted %int.make_type_signed.loc21_23, %.loc21_23.2 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc21: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc21: %i32 = bind_name n, %n.param.loc21
// CHECK:STDOUT:     %return.param.loc21: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc25_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_13.3: type = converted %int.make_type_signed, %.loc25_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_11: init type = call constants.%Int(%.loc12_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.2: type = value_of_initializer %int.make_type_signed.loc12_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.3: type = converted %int.make_type_signed.loc12_11, %.loc12_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_19: init type = call constants.%Int(%.loc12_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_19.2: type = value_of_initializer %int.make_type_signed.loc12_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_19.3: type = converted %int.make_type_signed.loc12_19, %.loc12_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc16_11: init type = call constants.%Int(%.loc16_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_11.2: type = value_of_initializer %int.make_type_signed.loc16_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_11.3: type = converted %int.make_type_signed.loc16_11, %.loc16_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc16_19: init type = call constants.%Int(%.loc16_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.2: type = value_of_initializer %int.make_type_signed.loc16_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.3: type = converted %int.make_type_signed.loc16_19, %.loc16_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc16: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc16: %i32 = bind_name n, %n.param.loc16
// CHECK:STDOUT:     %return.param.loc16: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %.loc16_16: ref %i32 = return_slot %return.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc18_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_10.3: type = converted %int.make_type_signed, %.loc18_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_8: %.2 = field_decl k, element0 [template]
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .k = %.loc18_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n.loc21
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc26_18.1: Core.IntLiteral = int_value 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_18.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_18.3: <bound method> = bound_method %.loc26_18.1, %.loc26_18.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc26_18.4: <specific function> = specific_function %.loc26_18.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc26_18.4(%.loc26_18.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc26_18.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   %.loc26_18.6: %i32 = converted %.loc26_18.1, %.loc26_18.5 [template = constants.%.33]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc26_18.6)
// CHECK:STDOUT:   %.loc26_20.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_20.2: %i32 = converted %F.call, %.loc26_20.1
// CHECK:STDOUT:   return %.loc26_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
