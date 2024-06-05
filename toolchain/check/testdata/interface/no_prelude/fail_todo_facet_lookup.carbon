// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_facet_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_facet_lookup.carbon

interface Interface { fn F(); }

fn CallStatic(T:! Interface) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+4]]:3: ERROR: Value of type `<associated <type of F> in Interface>` is not callable.
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
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Interface, <type of F> [template]
// CHECK:STDOUT:   %.4: <associated <type of F> in Interface> = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %T: Interface = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %CallStatic.type: type = fn_type @CallStatic [template]
// CHECK:STDOUT:   %CallStatic: <type of CallStatic> = struct_value () [template]
// CHECK:STDOUT:   %CallFacet.type: type = fn_type @CallFacet [template]
// CHECK:STDOUT:   %CallFacet: <type of CallFacet> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .CallStatic = %CallStatic.decl
// CHECK:STDOUT:     .CallFacet = %CallFacet.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %CallStatic.decl: <type of CallStatic> = fn_decl @CallStatic [template = constants.%CallStatic] {
// CHECK:STDOUT:     %Interface.ref.loc13: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc13_15.1: Interface = param T
// CHECK:STDOUT:     @CallStatic.%T: Interface = bind_symbolic_name T 0, %T.loc13_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFacet.decl: <type of CallFacet> = fn_decl @CallFacet [template = constants.%CallFacet] {
// CHECK:STDOUT:     %Interface.ref.loc21: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc21_14.1: Interface = param T
// CHECK:STDOUT:     @CallFacet.%T: Interface = bind_symbolic_name T 0, %T.loc21_14.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref: Interface = name_ref T, @CallFacet.%T [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc21_32.1: type = facet_type_access %T.ref [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc21_32.2: type = converted %T.ref, %.loc21_32.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.loc21_29.1: T = param x
// CHECK:STDOUT:     @CallFacet.%x: T = bind_name x, %x.loc21_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %.loc11: <associated <type of F> in Interface> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc11
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStatic(%T: Interface) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: Interface = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %F.ref: <associated <type of F> in Interface> = name_ref F, @Interface.%.loc11 [template = constants.%.4]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFacet(%T: Interface, %x: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: T = name_ref x, %x
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
