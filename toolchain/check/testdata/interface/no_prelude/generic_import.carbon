// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic_import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface AddWith(T:! type) {
  fn F();
}

// --- b.carbon

library "[[@TEST_NAME]]";
import library "a";

class C {}

impl C as AddWith(C) {
  fn F() {}
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.b35e: type = generic_interface_type @AddWith [template]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35e = struct_value () [template]
// CHECK:STDOUT:   %AddWith.type.c523: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.c523 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %AddWith.type.c523, %F.type [symbolic]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @AddWith.%F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.b35e = interface_decl @AddWith [template = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<invalid> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc4_19.2)> [symbolic = %AddWith.type (constants.%AddWith.type.c523)]
// CHECK:STDOUT:   %Self.2: %AddWith.type.c523 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T.loc4_19.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @AddWith.%AddWith.type (%AddWith.type.c523), @AddWith.%F.type (%F.type) [symbolic = %F.assoc_type (constants.%F.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @AddWith.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%AddWith.type (%AddWith.type.c523) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @AddWith.%F.type (%F.type) = fn_decl @F [symbolic = @AddWith.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @AddWith.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@AddWith.%T.loc4_19.1: type, @AddWith.%Self.1: @AddWith.%AddWith.type (%AddWith.type.c523)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(%T.loc4_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %AddWith.type.b35e: type = generic_interface_type @AddWith [template]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35e = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.b2db: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.b2db = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.debb: type = fn_type @F.1, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F.57ad: %F.type.debb = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.cbf7: type = assoc_entity_type %AddWith.type.b2db, %F.type.debb [symbolic]
// CHECK:STDOUT:   %assoc0.8166: %F.assoc_type.cbf7 = assoc_entity element0, imports.%import_ref.91c9 [symbolic]
// CHECK:STDOUT:   %AddWith.type.32bb: type = facet_type <@AddWith, @AddWith(%C)> [template]
// CHECK:STDOUT:   %F.type.47bd: type = fn_type @F.1, @AddWith(%C) [template]
// CHECK:STDOUT:   %F.0007: %F.type.47bd = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.13cd: type = assoc_entity_type %AddWith.type.32bb, %F.type.47bd [template]
// CHECK:STDOUT:   %assoc0.d296: %F.assoc_type.13cd = assoc_entity element0, imports.%import_ref.91c9 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.7447: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.7804: %F.type.7447 = struct_value () [template]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.b2db = facet_value %C, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.7c59: %AddWith.type.b35e = import_ref Main//a, AddWith, loaded [template = constants.%AddWith.generic]
// CHECK:STDOUT:   %import_ref.6486 = import_ref Main//a, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8a83 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %import_ref.70f0: @AddWith.%F.type (%F.type.debb) = import_ref Main//a, F, loaded [symbolic = @AddWith.%F (constants.%F.57ad)]
// CHECK:STDOUT:   %import_ref.91c9 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AddWith = imports.%import_ref.7c59
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.b35e = name_ref AddWith, imports.%import_ref.7c59 [template = constants.%AddWith.generic]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%C)> [template = constants.%AddWith.type.32bb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(constants.%T: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.b2db)]
// CHECK:STDOUT:   %Self: %AddWith.type.b2db = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @AddWith(%T) [symbolic = %F.type (constants.%F.type.debb)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type.debb) = struct_value () [symbolic = %F (constants.%F.57ad)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @AddWith.%AddWith.type (%AddWith.type.b2db), @AddWith.%F.type (%F.type.debb) [symbolic = %F.assoc_type (constants.%F.assoc_type.cbf7)]
// CHECK:STDOUT:   %assoc0: @AddWith.%F.assoc_type (%F.assoc_type.cbf7) = assoc_entity element0, imports.%import_ref.91c9 [symbolic = %assoc0 (constants.%assoc0.8166)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6486
// CHECK:STDOUT:     .F = imports.%import_ref.8a83
// CHECK:STDOUT:     witness = (imports.%import_ref.70f0)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref.loc7_6 as %AddWith.type {
// CHECK:STDOUT:   %F.decl: %F.type.7447 = fn_decl @F.2 [template = constants.%F.7804] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %AddWith.type.b2db) [from "a.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.32bb
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.47bd
// CHECK:STDOUT:   %F => constants.%F.0007
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.13cd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%AddWith.facet) {}
// CHECK:STDOUT:
