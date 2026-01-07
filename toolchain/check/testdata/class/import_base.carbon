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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_base.carbon

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
// CHECK:STDOUT:   %pattern_type.101: type = pattern_type %Base [concrete]
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
// CHECK:STDOUT:   %complete_type.cf1: <witness> = complete_type_witness %struct_type.x.unused_y [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.5a1: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.Unused.decl: %Base.Unused.type = fn_decl @Base.Unused [concrete = constants.%Base.Unused] {
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc8: %Base.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9: %Base.elem = field_decl unused_y, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.unused_y [concrete = constants.%complete_type.cf1]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.5a1]
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
// CHECK:STDOUT:   %struct_type.x.unused_y.9fc: type = struct_type {.x: %i32, .unused_y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %struct_type.x.unused_y.9fc [concrete]
// CHECK:STDOUT:   %struct_type.base.27a: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.5a1: <witness> = complete_type_witness %struct_type.base.27a [concrete]
// CHECK:STDOUT:   %pattern_type.454: type = pattern_type %Child [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y.76a: type = struct_type {.x: Core.IntLiteral, .unused_y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.0bf: %struct_type.x.unused_y.76a = struct_value (%int_0.5c6, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.base.503: type = struct_type {.base: %struct_type.x.unused_y.76a} [concrete]
// CHECK:STDOUT:   %struct.cb5: %struct_type.base.503 = struct_value (%struct.0bf) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.255: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.58d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e45, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.58d) [concrete]
// CHECK:STDOUT:   %.952: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.386: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9f1: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.263: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f07: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [concrete]
// CHECK:STDOUT:   %bound_method.307: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_0.263, %int_1.47b) [concrete]
// CHECK:STDOUT:   %Child.val: %Child = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1eb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [concrete]
// CHECK:STDOUT:   %bound_method.ef3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base = import_ref Main//a, Base, unloaded
// CHECK:STDOUT:   %Main.Child: type = import_ref Main//a, Child, loaded [concrete = constants.%Child]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc10_1, loaded [concrete = constants.%complete_type.2da]
// CHECK:STDOUT:   %Main.import_ref.691 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.062: %Base.F.type = import_ref Main//a, loc5_21, loaded [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Main.import_ref.7b8 = import_ref Main//a, loc6_26, unloaded
// CHECK:STDOUT:   %Main.import_ref.183: %Base.elem = import_ref Main//a, loc8_8, loaded [concrete = %.61a]
// CHECK:STDOUT:   %Main.import_ref.7c0 = import_ref Main//a, loc9_15, unloaded
// CHECK:STDOUT:   %Main.import_ref.44c: <witness> = import_ref Main//a, loc14_1, loaded [concrete = constants.%complete_type.5a1]
// CHECK:STDOUT:   %Main.import_ref.d37 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.4a6 = import_ref Main//a, loc13_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.bd3719.2: type = import_ref Main//a, loc13_16, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b25: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.255)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e45 = impl_witness_table (%Core.import_ref.b25), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %.61a: %Base.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = imports.%Main.Base
// CHECK:STDOUT:     .Child = imports.%Main.Child
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.44c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d37
// CHECK:STDOUT:   .base = imports.%Main.import_ref.4a6
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.bd3719.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.691
// CHECK:STDOUT:   .F = imports.%Main.import_ref.062
// CHECK:STDOUT:   .Unused = imports.%Main.import_ref.7b8
// CHECK:STDOUT:   .x = imports.%Main.import_ref.183
// CHECK:STDOUT:   .unused_y = imports.%Main.import_ref.7c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.454 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.454 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Child = var %a.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_49.1: %struct_type.x.unused_y.76a = struct_literal (%int_0, %int_1) [concrete = constants.%struct.0bf]
// CHECK:STDOUT:   %.loc7_50.1: %struct_type.base.503 = struct_literal (%.loc7_49.1) [concrete = constants.%struct.cb5]
// CHECK:STDOUT:   %impl.elem0.loc7_49.1: %.952 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:   %bound_method.loc7_49.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.386]
// CHECK:STDOUT:   %specific_fn.loc7_49.1: <specific function> = specific_function %impl.elem0.loc7_49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.2: <bound method> = bound_method %int_0, %specific_fn.loc7_49.1 [concrete = constants.%bound_method.9f1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1: init %i32 = call %bound_method.loc7_49.2(%int_0) [concrete = constants.%int_0.263]
// CHECK:STDOUT:   %.loc7_49.2: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1 [concrete = constants.%int_0.263]
// CHECK:STDOUT:   %.loc7_50.2: ref %Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_49.3: ref %i32 = class_element_access %.loc7_50.2, element0
// CHECK:STDOUT:   %.loc7_49.4: init %i32 to %.loc7_49.3 = initialize_from %.loc7_49.2 [concrete = constants.%int_0.263]
// CHECK:STDOUT:   %impl.elem0.loc7_49.2: %.952 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:   %bound_method.loc7_49.3: <bound method> = bound_method %int_1, %impl.elem0.loc7_49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f07]
// CHECK:STDOUT:   %specific_fn.loc7_49.2: <specific function> = specific_function %impl.elem0.loc7_49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.4: <bound method> = bound_method %int_1, %specific_fn.loc7_49.2 [concrete = constants.%bound_method.307]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2: init %i32 = call %bound_method.loc7_49.4(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc7_49.5: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc7_49.6: ref %i32 = class_element_access %.loc7_50.2, element1
// CHECK:STDOUT:   %.loc7_49.7: init %i32 to %.loc7_49.6 = initialize_from %.loc7_49.5 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc7_49.8: init %Base to %.loc7_50.2 = class_init (%.loc7_49.4, %.loc7_49.7) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_50.3: init %Base = converted %.loc7_49.1, %.loc7_49.8 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_50.4: init %Child to %a.var = class_init (%.loc7_50.3) [concrete = constants.%Child.val]
// CHECK:STDOUT:   %.loc7_3: init %Child = converted %.loc7_50.1, %.loc7_50.4 [concrete = constants.%Child.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, imports.%Main.Child [concrete = constants.%Child]
// CHECK:STDOUT:   %a: ref %Child = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%Main.import_ref.183 [concrete = imports.%.61a]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8: %.952 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:   %bound_method.loc8_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1eb]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_7.2: <bound method> = bound_method %int_2, %specific_fn.loc8 [concrete = constants.%bound_method.ef3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_7.2(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:   %.loc8_7: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, imports.%Main.import_ref.062 [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_3.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.2: ref %Base = converted %a.ref.loc9, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.3: %Base = acquire_value %.loc9_3.2
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %Base.F.bound(%.loc9_3.3)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Child) = "no_op";
// CHECK:STDOUT:
