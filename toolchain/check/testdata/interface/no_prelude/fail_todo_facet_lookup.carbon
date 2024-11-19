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
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+4]]:3: error: value of type `<associated <type of F> in Interface>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   T.F();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.F();
}

fn CallFacet(T:! Interface, x: T) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+3]]:3: error: type `T` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   x.F();
  // CHECK:STDERR:   ^~~
  x.F();
}

// CHECK:STDOUT: --- fail_todo_facet_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [template]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Interface.type, %F.type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %T: %Interface.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Interface.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CallStatic.type: type = fn_type @CallStatic [template]
// CHECK:STDOUT:   %CallStatic: %CallStatic.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = facet_access_type %T [symbolic]
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
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %CallStatic.decl: %CallStatic.type = fn_decl @CallStatic [template = constants.%CallStatic] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %Interface.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %Interface.type = value_param_pattern %T.patt.loc13_15.1, runtime_param<invalid> [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %Interface.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_15.1: %Interface.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFacet.decl: %CallFacet.type = fn_decl @CallFacet [template = constants.%CallFacet] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: %Interface.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %Interface.type = value_param_pattern %T.patt.loc21_14.1, runtime_param<invalid> [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallFacet.%.loc21_32.3 (%.3) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallFacet.%.loc21_32.3 (%.3) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.ref: %Interface.type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %.loc21_32.1: type = facet_access_type %T.ref [symbolic = %.loc21_32.3 (constants.%.3)]
// CHECK:STDOUT:     %.loc21_32.2: type = converted %T.ref, %.loc21_32.1 [symbolic = %.loc21_32.3 (constants.%.3)]
// CHECK:STDOUT:     %T.param: %Interface.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc21_14.1: %Interface.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallFacet.%.loc21_32.3 (%.3) = value_param runtime_param0
// CHECK:STDOUT:     %x: @CallFacet.%.loc21_32.3 (%.3) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc11: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc11
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallStatic(%T.loc13_15.1: %Interface.type) {
// CHECK:STDOUT:   %T.loc13_15.2: %Interface.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_15.2: %Interface.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Interface.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %Interface.type = name_ref T, %T.loc13_15.1 [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref: %.1 = name_ref F, @Interface.%.loc11 [template = constants.%.2]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallFacet(%T.loc21_14.1: %Interface.type) {
// CHECK:STDOUT:   %T.loc21_14.2: %Interface.type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: %Interface.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc21_32.3: type = facet_access_type %T.loc21_14.2 [symbolic = %.loc21_32.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %Interface.type, %x.param_patt: @CallFacet.%.loc21_32.3 (%.3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @CallFacet.%.loc21_32.3 (%.3) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallStatic(constants.%T) {
// CHECK:STDOUT:   %T.loc13_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallFacet(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %.loc21_32.3 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
