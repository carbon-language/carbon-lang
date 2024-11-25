// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/todo_access_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/todo_access_modifiers.carbon

// TODO: Test calls to these (member access control is not yet implemented).
class Access {
  private fn F();

  protected fn G();

  private var k: i32;

  protected var l: i32;
}

// CHECK:STDOUT: --- todo_access_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Access: type = class_type @Access [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Access, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: %i32, .l: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
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
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Access.decl: type = class_decl @Access [template = constants.%Access] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Access {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %.loc17_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_18.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_18.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_18.3: type = converted %int.make_type_signed.loc17, %.loc17_18.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_16: %.2 = field_decl k, element0 [template]
// CHECK:STDOUT:   %.loc19_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_20.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_20.2: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_20.3: type = converted %int.make_type_signed.loc19, %.loc19_20.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_18: %.2 = field_decl l, element1 [template]
// CHECK:STDOUT:   %.loc20: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Access
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G [protected] = %G.decl
// CHECK:STDOUT:   .k [private] = %.loc17_16
// CHECK:STDOUT:   .l [protected] = %.loc19_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
