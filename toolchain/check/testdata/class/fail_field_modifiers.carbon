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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %int.0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.17: <bound method> = bound_method %int.0.1, %Convert.14 [template]
// CHECK:STDOUT:   %.18: <specific function> = specific_function %.17, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.20: <specific function> = specific_function %.19, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [template]
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
// CHECK:STDOUT:   %int.32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int.32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_18.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_18.2: type = converted %int.make_type_signed.loc17, %.loc17_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_16: %.1 = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.32.loc23: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%int.32.loc23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_16.1: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_16.2: type = converted %int.make_type_signed.loc23, %.loc23_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_14: %.1 = field_decl k, element1 [template]
// CHECK:STDOUT:   %int.32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%int.32.loc29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_18.1: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_18.2: type = converted %int.make_type_signed.loc29, %.loc29_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_25.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_25.2: <bound method> = bound_method %int.0, %.loc29_25.1 [template = constants.%.17]
// CHECK:STDOUT:   %.loc29_25.3: <specific function> = specific_function %.loc29_25.2, @Convert.2(constants.%int.32) [template = constants.%.18]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %.loc29_25.3(%int.0) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc29_25.4: %i32 = value_of_initializer %int.convert_checked.loc29 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc29_25.5: %i32 = converted %int.0, %.loc29_25.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %l: %i32 = bind_name l, %.loc29_25.5
// CHECK:STDOUT:   %int.32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%int.32.loc34) [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_16.1: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_16.2: type = converted %int.make_type_signed.loc34, %.loc34_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc34_23.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_23.2: <bound method> = bound_method %int.1, %.loc34_23.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc34_23.3: <specific function> = specific_function %.loc34_23.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc34: init %i32 = call %.loc34_23.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc34_23.4: %i32 = value_of_initializer %int.convert_checked.loc34 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc34_23.5: %i32 = converted %int.1, %.loc34_23.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %m: %i32 = bind_name m, %.loc34_23.5
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc17_16
// CHECK:STDOUT:   .k = %.loc23_14
// CHECK:STDOUT:   .l = %l
// CHECK:STDOUT:   .m = %m
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
