// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn F() {}
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F() {}

  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32 = "int.sadd";
  // CHECK:STDERR:   ^~~~~~~
  default fn G(a: i32, b: i32) -> i32 = "int.sadd";
}

// CHECK:STDOUT: --- fail_todo_define_default_fn_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Interface, F [template]
// CHECK:STDOUT:   %.4: <associated F in Interface> = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Interface, G [template]
// CHECK:STDOUT:   %.6: <associated G in Interface> = assoc_entity element1, @Interface.%G.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc12: <associated F in Interface> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc17_16.1: i32 = param a
// CHECK:STDOUT:     %a.loc17_16.2: i32 = bind_name a, %a.loc17_16.1
// CHECK:STDOUT:     %b.loc17_24.1: i32 = param b
// CHECK:STDOUT:     %b.loc17_24.2: i32 = bind_name b, %b.loc17_24.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <associated G in Interface> = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   .G = %.loc17
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@Interface.%a.loc17_16.2: i32, @Interface.%b.loc17_24.2: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
