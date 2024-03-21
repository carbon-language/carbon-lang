// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface { fn F(); }

fn CallStatic(T:! Interface) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+4]]:3: ERROR: Value of type `<associated <function> in Interface>` is not callable.
  // CHECK:STDERR:   T.F();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.F();
}

fn CallFacet(T:! Interface, x: T) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+3]]:3: ERROR: Type `T` does not support qualified expressions.
  // CHECK:STDERR:   x.F();
  // CHECK:STDERR:   ^~~
  x.F();
}

// CHECK:STDOUT: --- fail_todo_facet_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Interface, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Interface> = assoc_entity element0, @Interface.%F [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .CallStatic = %CallStatic
// CHECK:STDOUT:     .CallFacet = %CallFacet
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %CallStatic: <function> = fn_decl @CallStatic [template] {
// CHECK:STDOUT:     %Interface.ref.loc9: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc9_15.1: Interface = param T
// CHECK:STDOUT:     @CallStatic.%T: Interface = bind_symbolic_name T, %T.loc9_15.1 [symbolic]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFacet: <function> = fn_decl @CallFacet [template] {
// CHECK:STDOUT:     %Interface.ref.loc17: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc17_14.1: Interface = param T
// CHECK:STDOUT:     @CallFacet.%T: Interface = bind_symbolic_name T, %T.loc17_14.1 [symbolic]
// CHECK:STDOUT:     %T.ref: Interface = name_ref T, @CallFacet.%T [symbolic = @CallFacet.%T]
// CHECK:STDOUT:     %.loc17_32.1: type = facet_type_access %T.ref [symbolic = @CallFacet.%T]
// CHECK:STDOUT:     %.loc17_32.2: type = converted %T.ref, %.loc17_32.1 [symbolic = @CallFacet.%T]
// CHECK:STDOUT:     %x.loc17_29.1: T = param x
// CHECK:STDOUT:     @CallFacet.%x: T = bind_name x, %x.loc17_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %.loc7: <associated <function> in Interface> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc7
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStatic(%T: Interface) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: Interface = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %F.ref: <associated <function> in Interface> = name_ref F, @Interface.%.loc7 [template = constants.%.3]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFacet(%T: Interface, %x: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: T = name_ref x, %x
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
