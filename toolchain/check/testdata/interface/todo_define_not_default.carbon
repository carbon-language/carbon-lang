// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  // TODO: A definition without `default` in an interface should be rejected.
  fn F() {}
  fn G(a: i32, b: i32) -> i32 = "int.add";

  // TODO: An associated constant with an initializer without `default` in an
  // interface should be rejected.
  let T:! type = (i32, i32);
  let N:! i32 = 42;
}

// CHECK:STDOUT: --- todo_define_not_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in I> = assoc_entity element0, @I.%F [template]
// CHECK:STDOUT:   %.4: <associated <function> in I> = assoc_entity element1, @I.%G [template]
// CHECK:STDOUT:   %.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.8: <associated type in I> = assoc_entity element2, @I.%T [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type @I, i32 [template]
// CHECK:STDOUT:   %.11: <associated i32 in I> = assoc_entity element3, @I.%N [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %.loc9: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %a.loc10_8.1: i32 = param a
// CHECK:STDOUT:     %a.loc10_8.2: i32 = bind_name a, %a.loc10_8.1
// CHECK:STDOUT:     %b.loc10_16.1: i32 = param b
// CHECK:STDOUT:     %b.loc10_16.2: i32 = bind_name b, %b.loc10_16.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <associated <function> in I> = assoc_entity element1, %G [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_27.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc14_27.2: type = converted %.loc14_27.1, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc14_28: <associated type in I> = assoc_entity element2, %T [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_17: i32 = int_literal 42 [template = constants.%.9]
// CHECK:STDOUT:   %N: i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc15_19: <associated i32 in I> = assoc_entity element3, %N [template = constants.%.11]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc9
// CHECK:STDOUT:   .G = %.loc10
// CHECK:STDOUT:   .T = %.loc14_28
// CHECK:STDOUT:   .N = %.loc15_19
// CHECK:STDOUT:   witness = (%F, %G, %T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@I.%a.loc10_8.2: i32, @I.%b.loc10_16.2: i32) -> i32 = "int.add";
// CHECK:STDOUT:
