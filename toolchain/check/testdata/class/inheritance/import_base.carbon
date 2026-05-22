// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/import_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/import_base.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn F[self: Self]();
  fn Unused[self: Self]();

  var x: i32;
  var unused_y: i32;
}

class Child {
  extend base: Base;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Child = {.base = {.x = 0, .unused_y = 1}};
  a.x = 2;
  a.F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.Unused.type: type = fn_type @Base.Unused [concrete]
// CHECK:STDOUT:   %Base.Unused: %Base.Unused.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y: type = struct_type {.x: %i32, .unused_y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf9: <witness> = complete_type_witness %struct_type.x.unused_y [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.3b0: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Child = %Child.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Child.decl: type = class_decl @Child [concrete = constants.%Child] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.fa7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.Unused.decl: %Base.Unused.type = fn_decl @Base.Unused [concrete = constants.%Base.Unused] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.fa7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: %Base.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %.loc9: %Base.elem = field_decl unused_y, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.unused_y [concrete = constants.%complete_type.cf9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT:   .Unused = %Base.Unused.decl
// CHECK:STDOUT:   .x = %.loc8
// CHECK:STDOUT:   .unused_y = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc13: %Child.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.3b0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Child
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.Unused(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y.b4b: type = struct_type {.x: %i32, .unused_y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.e4d: <witness> = complete_type_witness %struct_type.x.unused_y.b4b [concrete]
// CHECK:STDOUT:   %struct_type.base.1b1: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.043: <witness> = complete_type_witness %struct_type.base.1b1 [concrete]
// CHECK:STDOUT:   %pattern_type.710: type = pattern_type %Child [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y.76a: type = struct_type {.x: Core.IntLiteral, .unused_y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.0bf: %struct_type.x.unused_y.76a = struct_value (%int_0.5c6, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.base.503: type = struct_type {.base: %struct_type.x.unused_y.76a} [concrete]
// CHECK:STDOUT:   %struct.cb5: %struct_type.base.503 = struct_value (%struct.0bf) [concrete]
// CHECK:STDOUT:   %.9de: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.56e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ca9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.768: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ca9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1e6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bc3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.108: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.924: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.108 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.56e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1e6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.68b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.634: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.68b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1dd: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.924 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.924, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.296: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.d70: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.924 [concrete]
// CHECK:STDOUT:   %bound_method.b34: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.50a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.28c: %.9de = struct_value (%int_0.d70, %int_1.50a) [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_0.d70, %int_1.50a) [concrete]
// CHECK:STDOUT:   %Child.val: %Child = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f4c: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.924 [concrete]
// CHECK:STDOUT:   %bound_method.fe7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.21f: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc7_3.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base = import_ref Main//a, Base, unloaded
// CHECK:STDOUT:   %Main.Child: type = import_ref Main//a, Child, loaded [concrete = constants.%Child]
// CHECK:STDOUT:   %Core.b38: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7f9: <witness> = import_ref Main//a, loc10_1, loaded [concrete = constants.%complete_type.e4d]
// CHECK:STDOUT:   %Main.import_ref.e52 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e20: %Base.F.type = import_ref Main//a, loc5_21, loaded [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Main.import_ref.384 = import_ref Main//a, loc6_26, unloaded
// CHECK:STDOUT:   %Main.import_ref.fc2: %Base.elem = import_ref Main//a, loc8_8, loaded [concrete = %.98f]
// CHECK:STDOUT:   %Main.import_ref.4cb = import_ref Main//a, loc9_15, unloaded
// CHECK:STDOUT:   %Main.import_ref.e2d: <witness> = import_ref Main//a, loc14_1, loaded [concrete = constants.%complete_type.043]
// CHECK:STDOUT:   %Main.import_ref.ffc = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.2a5 = import_ref Main//a, loc13_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.ac35b2.2: type = import_ref Main//a, loc13_16, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7f1: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ca9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.768)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bc3 = impl_witness_table (%Core.import_ref.7f1), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %.98f: %Base.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = imports.%Main.Base
// CHECK:STDOUT:     .Child = imports.%Main.Child
// CHECK:STDOUT:     .Core = imports.%Core.b38
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.e2d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ffc
// CHECK:STDOUT:   .base = imports.%Main.import_ref.2a5
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.ac35b2.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.7f9
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e52
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e20
// CHECK:STDOUT:   .Unused = imports.%Main.import_ref.384
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fc2
// CHECK:STDOUT:   .unused_y = imports.%Main.import_ref.4cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.710 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.710 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Child = var %a.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_49.1: %struct_type.x.unused_y.76a = struct_literal (%int_0, %int_1) [concrete = constants.%struct.0bf]
// CHECK:STDOUT:   %.loc7_50.1: %struct_type.base.503 = struct_literal (%.loc7_49.1) [concrete = constants.%struct.cb5]
// CHECK:STDOUT:   %impl.elem0.loc7_49.1: %.634 = impl_witness_access constants.%ImplicitAs.impl_witness.1e6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.924]
// CHECK:STDOUT:   %bound_method.loc7_49.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1dd]
// CHECK:STDOUT:   %specific_fn.loc7_49.1: <specific function> = specific_function %impl.elem0.loc7_49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.2: <bound method> = bound_method %int_0, %specific_fn.loc7_49.1 [concrete = constants.%bound_method.296]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1: init %i32 = call %bound_method.loc7_49.2(%int_0) [concrete = constants.%int_0.d70]
// CHECK:STDOUT:   %.loc7_49.2: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1 [concrete = constants.%int_0.d70]
// CHECK:STDOUT:   %.loc7_50.2: ref %.9de = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_49.3: ref %i32 = class_element_access %.loc7_50.2, element0
// CHECK:STDOUT:   %.loc7_49.4: init %i32 to %.loc7_49.3 = in_place_init %.loc7_49.2 [concrete = constants.%int_0.d70]
// CHECK:STDOUT:   %impl.elem0.loc7_49.2: %.634 = impl_witness_access constants.%ImplicitAs.impl_witness.1e6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.924]
// CHECK:STDOUT:   %bound_method.loc7_49.3: <bound method> = bound_method %int_1, %impl.elem0.loc7_49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff4]
// CHECK:STDOUT:   %specific_fn.loc7_49.2: <specific function> = specific_function %impl.elem0.loc7_49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.4: <bound method> = bound_method %int_1, %specific_fn.loc7_49.2 [concrete = constants.%bound_method.b34]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2: init %i32 = call %bound_method.loc7_49.4(%int_1) [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc7_49.5: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc7_49.6: ref %i32 = class_element_access %.loc7_50.2, element1
// CHECK:STDOUT:   %.loc7_49.7: init %i32 to %.loc7_49.6 = in_place_init %.loc7_49.5 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc7_49.8: init %.9de to %.loc7_50.2 = class_init (%.loc7_49.4, %.loc7_49.7) [concrete = constants.%struct.28c]
// CHECK:STDOUT:   %.loc7_50.3: init %.9de = converted %.loc7_49.1, %.loc7_49.8 [concrete = constants.%struct.28c]
// CHECK:STDOUT:   %.loc7_50.4: init %Base = as_compatible %.loc7_50.3 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_50.5: init %Child to %a.var = class_init (%.loc7_50.4) [concrete = constants.%Child.val]
// CHECK:STDOUT:   %.loc7_3: init %Child = converted %.loc7_50.1, %.loc7_50.5 [concrete = constants.%Child.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, imports.%Main.Child [concrete = constants.%Child]
// CHECK:STDOUT:   %a: ref %Child = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%Main.import_ref.fc2 [concrete = imports.%.98f]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8: %.634 = impl_witness_access constants.%ImplicitAs.impl_witness.1e6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.924]
// CHECK:STDOUT:   %bound_method.loc8_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f4c]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_7.2: <bound method> = bound_method %int_2, %specific_fn.loc8 [concrete = constants.%bound_method.fe7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_7.2(%int_2) [concrete = constants.%int_2.21f]
// CHECK:STDOUT:   %.loc8_7: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_2.21f]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, imports.%Main.import_ref.e20 [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_3.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.2: ref %Base = converted %a.ref.loc9, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.3: %Base = acquire_value %.loc9_3.2
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %Base.F.bound(%.loc9_3.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.6
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %struct_type.x.unused_y.b4b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.4(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.5(%self.param: ref %struct_type.base.1b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.6(%self.param: ref %Child) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
