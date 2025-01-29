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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Access.elem: type = unbound_element_type %Access, %i32 [template]
// CHECK:STDOUT:   %struct_type.k.l: type = struct_type {.k: %i32, .l: %i32} [template]
// CHECK:STDOUT:   %complete_type.48a: <witness> = complete_type_witness %struct_type.k.l [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
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
// CHECK:STDOUT:   %.loc17_16: %Access.elem = field_decl k, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc17_11: %Access.elem = var_pattern %.loc17_16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc17: ref %Access.elem = var <none>
// CHECK:STDOUT:   %.loc19_18: %Access.elem = field_decl l, element1 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc19_13: %Access.elem = var_pattern %.loc19_18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc19: ref %Access.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k.l [template = constants.%complete_type.48a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
