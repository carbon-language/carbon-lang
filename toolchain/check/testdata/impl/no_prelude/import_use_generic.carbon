// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_use_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_use_generic.carbon

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {}

interface I {
  fn F();
}

impl forall [T:! type] C(T) as I {
  fn F() {}
}

// --- fail_use_in_fn_decl.carbon

library "[[@TEST_NAME]]";

import library "import_generic";

var c: C({}) = {};
// We're just checking that this doesn't crash. It's not expected to compile.
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+7]]:11: error: package `Core` implicitly referenced here, but not found [CoreNotFound]
// CHECK:STDERR: fn F() -> c.(I.F)() {}
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+3]]:22: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: fn F() -> c.(I.F)() {}
// CHECK:STDERR:                      ^
fn F() -> c.(I.F)() {}

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: %I.type = facet_value %C.2, %C.2 [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_14.1, runtime_param<invalid> [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_25.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_25.2 (constants.%C.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc7: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc7
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc10_14.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc10_25.2: type = class_type @C, @C(%T.loc10_14.2) [symbolic = %C.loc10_25.2 (constants.%C.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc10_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc10_34.2: <witness> = interface_witness (%F) [symbolic = %.loc10_34.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_25.1 as %I.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc10_34.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc10_34.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc10_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc10_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc10_14.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_25.2 => constants.%C.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc10_34.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc10_14.2) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_25.2 => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_in_fn_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.1) [template]
// CHECK:STDOUT:   %struct: %C.3 = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%.1) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %F.3, @F.2(%.1) [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref Main//import_generic, inst+7, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//import_generic, inst+17, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//import_generic, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//import_generic, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref Main//import_generic, inst+25, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//import_generic, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Main//import_generic, inst+32, loaded [symbolic = @impl.%C (constants.%C.2)]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Main//import_generic, inst+33, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Main//import_generic, inst+43, loaded [symbolic = @impl.%.1 (constants.%.6)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.1 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc6_11: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc6_9: type = converted %.loc6_11, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%.1) [template = constants.%C.3]
// CHECK:STDOUT:   %c.var: ref %C.3 = var c
// CHECK:STDOUT:   %c: ref %C.3 = bind_name c, %c.var
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.3 [template = constants.%F.4] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.ref: ref %C.3 = name_ref c, file.%c
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.2 [template = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_12.1: %F.type.1 = interface_witness_access constants.%.7, element0 [template = constants.%F.3]
// CHECK:STDOUT:     %.loc15_12.2: <specific function> = specific_function %.loc15_12.1, @F.2(constants.%.1) [template = constants.%.8]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %.loc15_12.2()
// CHECK:STDOUT:     %.loc15_18.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:     %.loc15_18.2: ref %empty_tuple.type = temporary %.loc15_18.1, %F.call
// CHECK:STDOUT:     %.loc15_19: type = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .F = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(constants.%T: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_17.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc6_17.2: init %C.3 = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_18: init %C.3 = converted %.loc6_17.1, %.loc6_17.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.1) {
// CHECK:STDOUT:   %T => constants.%.1
// CHECK:STDOUT:   %T.patt => constants.%.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.1) {
// CHECK:STDOUT:   %T => constants.%.1
// CHECK:STDOUT:   %T.patt => constants.%.1
// CHECK:STDOUT:   %C => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
