// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   default var j: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   final var k: i32;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~
  final let m: i32 = 1;
}

// CHECK:STDOUT: --- fail_field_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %.34: type = struct_type {.j: %.2, .k: %.2} [template]
// CHECK:STDOUT:   %.35: <witness> = complete_type_witness %.34 [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc17_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_18.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_18.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_18.3: type = converted %int.make_type_signed.loc17, %.loc17_18.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_16: %.3 = field_decl j, element0 [template]
// CHECK:STDOUT:   %.loc23_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%.loc23_16.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_16.2: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_16.3: type = converted %int.make_type_signed.loc23, %.loc23_16.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_14: %.3 = field_decl k, element1 [template]
// CHECK:STDOUT:   %.loc29_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%.loc29_18.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_18.2: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_18.3: type = converted %int.make_type_signed.loc29, %.loc29_18.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_24: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc29_25.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_25.2: <bound method> = bound_method %.loc29_24, %.loc29_25.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc29_25.3: <specific function> = specific_function %.loc29_25.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %.2 = call %.loc29_25.3(%.loc29_24) [template = constants.%.29]
// CHECK:STDOUT:   %.loc29_25.4: %.2 = value_of_initializer %int.convert_checked.loc29 [template = constants.%.29]
// CHECK:STDOUT:   %.loc29_25.5: %.2 = converted %.loc29_24, %.loc29_25.4 [template = constants.%.29]
// CHECK:STDOUT:   %l: %.2 = bind_name l, %.loc29_25.5
// CHECK:STDOUT:   %.loc34_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%.loc34_16.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_16.2: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_16.3: type = converted %int.make_type_signed.loc34, %.loc34_16.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_22: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc34_23.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_23.2: <bound method> = bound_method %.loc34_22, %.loc34_23.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc34_23.3: <specific function> = specific_function %.loc34_23.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc34: init %.2 = call %.loc34_23.3(%.loc34_22) [template = constants.%.33]
// CHECK:STDOUT:   %.loc34_23.4: %.2 = value_of_initializer %int.convert_checked.loc34 [template = constants.%.33]
// CHECK:STDOUT:   %.loc34_23.5: %.2 = converted %.loc34_22, %.loc34_23.4 [template = constants.%.33]
// CHECK:STDOUT:   %m: %.2 = bind_name m, %.loc34_23.5
// CHECK:STDOUT:   %.loc35: <witness> = complete_type_witness %.34 [template = constants.%.35]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc17_16
// CHECK:STDOUT:   .k = %.loc23_14
// CHECK:STDOUT:   .l = %l
// CHECK:STDOUT:   .m = %m
// CHECK:STDOUT: }
// CHECK:STDOUT:
