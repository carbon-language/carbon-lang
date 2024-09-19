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
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+4]]:3: error: value of type `<associated <type of F> in Interface>` is not callable
  // CHECK:STDERR:   T.F();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.F();
}

fn CallFacet(T:! Interface, x: T) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+3]]:3: error: type `T` does not support qualified expressions
  // CHECK:STDERR:   x.F();
  // CHECK:STDERR:   ^~~
  x.F();
}

// CHECK:STDOUT: --- fail_todo_facet_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %T: %.1 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %CallStatic.type: type = fn_type @CallStatic [template]
// CHECK:STDOUT:   %CallStatic: %CallStatic.type = struct_value () [template]
// CHECK:STDOUT:   %CallFacet.type: type = fn_type @CallFacet [template]
// CHECK:STDOUT:   %CallFacet: %CallFacet.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .CallStatic = %CallStatic.decl
// CHECK:STDOUT:     .CallFacet = %CallFacet.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %CallStatic.decl: %CallStatic.type = fn_decl @CallStatic [template = constants.%CallStatic] {
// CHECK:STDOUT:     %Interface.ref.loc13: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc13_15.1: %.1 = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallStatic.%T.loc13: %.1 = bind_symbolic_name T 0, %T.loc13_15.1 [symbolic = @CallStatic.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFacet.decl: %CallFacet.type = fn_decl @CallFacet [template = constants.%CallFacet] {
// CHECK:STDOUT:     %Interface.ref.loc21: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc21_14.1: %.1 = param T, runtime_param<invalid>
// CHECK:STDOUT:     @CallFacet.%T.loc21: %.1 = bind_symbolic_name T 0, %T.loc21_14.1 [symbolic = @CallFacet.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, @CallFacet.%T.loc21 [symbolic = @CallFacet.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21_32.1: type = facet_type_access %T.ref [symbolic = @CallFacet.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21_32.2: type = converted %T.ref, %.loc21_32.1 [symbolic = @CallFacet.%T.1 (constants.%T)]
// CHECK:STDOUT:     %x.loc21_29.1: @CallFacet.%T.1 (%T) = param x, runtime_param0
// CHECK:STDOUT:     @CallFacet.%x: @CallFacet.%T.1 (%T) = bind_name x, %x.loc21_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %.loc11: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc11
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallStatic(%T.loc13: %.1) {
// CHECK:STDOUT:   %T.1: %.1 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc13: %.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %F.ref: %.3 = name_ref F, @Interface.%.loc11 [template = constants.%.4]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallFacet(%T.loc21: %.1) {
// CHECK:STDOUT:   %T.1: %.1 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc21: %.1, %x: @CallFacet.%T.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @CallFacet.%T.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallStatic(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallFacet(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
