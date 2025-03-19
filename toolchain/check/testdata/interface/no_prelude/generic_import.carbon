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
// CHECK:STDOUT:   %AddWith.type.b35: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.bc7: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.bc7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type %AddWith.type.bc7 [symbolic]
// CHECK:STDOUT:   %assoc0: %AddWith.assoc_type = assoc_entity element0, @AddWith.%F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.b35 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc4_19.2)> [symbolic = %AddWith.type (constants.%AddWith.type.bc7)]
// CHECK:STDOUT:   %Self.2: %AddWith.type.bc7 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @AddWith(%T.loc4_19.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith.%AddWith.type (%AddWith.type.bc7) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%AddWith.type (%AddWith.type.bc7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @AddWith.%F.type (%F.type) = fn_decl @F [symbolic = @AddWith.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@AddWith.%T.loc4_19.1: type, @AddWith.%Self.1: @AddWith.%AddWith.type (%AddWith.type.bc7)) {
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AddWith.type.b35: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.bc7: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %AddWith.type.bc7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.fbc: type = fn_type @F.1, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %F.be3: %F.type.fbc = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.73f: type = assoc_entity_type %AddWith.type.bc7 [symbolic]
// CHECK:STDOUT:   %assoc0.80d: %AddWith.assoc_type.73f = assoc_entity element0, imports.%Main.import_ref.0c5 [symbolic]
// CHECK:STDOUT:   %AddWith.type.e8e: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %F.type.cd2: type = fn_type @F.1, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %F.b58: %F.type.cd2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.ace: type = assoc_entity_type %AddWith.type.e8e [concrete]
// CHECK:STDOUT:   %assoc0.a9f: %AddWith.assoc_type.ace = assoc_entity element0, imports.%Main.import_ref.0c5 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.c09: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.e62: %F.type.c09 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.e8e = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.AddWith: %AddWith.type.b35 = import_ref Main//a, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//a, loc4_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.476 = import_ref Main//a, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.550 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT:   %Main.F: @AddWith.%F.type (%F.type.fbc) = import_ref Main//a, F, loaded [symbolic = @AddWith.%F (constants.%F.be3)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//a, loc4_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5a4: @AddWith.%AddWith.type (%AddWith.type.bc7) = import_ref Main//a, inst25 [no loc], loaded [symbolic = @AddWith.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.0c5 = import_ref Main//a, loc5_9, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AddWith = imports.%Main.AddWith
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.b35 = name_ref AddWith, imports.%Main.AddWith [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %C.ref.loc7_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%C)> [concrete = constants.%AddWith.type.e8e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Main.import_ref.5ab3ec.1: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.bc7)]
// CHECK:STDOUT:   %Self: %AddWith.type.bc7 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @AddWith(%T) [symbolic = %F.type (constants.%F.type.fbc)]
// CHECK:STDOUT:   %F: @AddWith.%F.type (%F.type.fbc) = struct_value () [symbolic = %F (constants.%F.be3)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith.%AddWith.type (%AddWith.type.bc7) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.73f)]
// CHECK:STDOUT:   %assoc0: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.73f) = assoc_entity element0, imports.%Main.import_ref.0c5 [symbolic = %assoc0 (constants.%assoc0.80d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.476
// CHECK:STDOUT:     .F = imports.%Main.import_ref.550
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref.loc7_6 as %AddWith.type {
// CHECK:STDOUT:   %F.decl: %F.type.c09 = fn_decl @F.2 [concrete = constants.%F.e62] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.5a4: @AddWith.%AddWith.type (%AddWith.type.bc7)) [from "a.carbon"] {
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
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.e8e
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.cd2
// CHECK:STDOUT:   %F => constants.%F.b58
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.ace
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%C, constants.%AddWith.facet) {}
// CHECK:STDOUT:
