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
  fn F(self);
  fn Unused(self);

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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.912: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.912 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.912 = wrapper_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.Unused.type: type = fn_type @Base.Unused [concrete]
// CHECK:STDOUT:   %Base.Unused: %Base.Unused.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y: type = struct_type {.x: %i32, .unused_y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.d98: <witness> = complete_type_witness %struct_type.x.unused_y [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.c02: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.912 = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.912 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.Unused.decl: %Base.Unused.type = fn_decl @Base.Unused [concrete = constants.%Base.Unused] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.912 = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.912 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl.loc8: %Base.elem = field_decl x, element0, %i32.loc8 in [concrete] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl.loc9: %Base.elem = field_decl unused_y, element1, %i32.loc9 in [concrete] {
// CHECK:STDOUT:     %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.unused_y [concrete = constants.%complete_type.d98]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT:   .Unused = %Base.Unused.decl
// CHECK:STDOUT:   .x = %field_decl.loc8
// CHECK:STDOUT:   .unused_y = %field_decl.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc13: %Child.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.c02]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y.20e: type = struct_type {.x: %i32, .unused_y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.d98: <witness> = complete_type_witness %struct_type.x.unused_y.20e [concrete]
// CHECK:STDOUT:   %Child.elem: type = unbound_element_type %Child, %Base [concrete]
// CHECK:STDOUT:   %struct_type.base.cae: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.c02: <witness> = complete_type_witness %struct_type.base.cae [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %Child [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.860 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.860 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x.unused_y.76a: type = struct_type {.x: Core.IntLiteral, .unused_y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.0bf: %struct_type.x.unused_y.76a = struct_value (%int_0.5c6, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.base.503: type = struct_type {.base: %struct_type.x.unused_y.76a} [concrete]
// CHECK:STDOUT:   %struct.cb5: %struct_type.base.503 = struct_value (%struct.0bf) [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.be3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.bd4: %.edf = struct_value (%int_0.1b1, %int_1.874) [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_0.1b1, %int_1.874) [concrete]
// CHECK:STDOUT:   %Child.val: %Child = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6df: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6df) [concrete]
// CHECK:STDOUT:   %facet_value.c23: %facet_type = facet_value %i32, (%DestroyImpl.impl_witness.52c, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.da9: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.c23) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c7: %SubobjectDestroy.type = facet_value %Base, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.69a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c7) [concrete]
// CHECK:STDOUT:   %facet_value.0ce: %facet_type = facet_value %Base, (%DestroyImpl.impl_witness.69a, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.de2: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.0ce) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9ab: %SubobjectDestroy.type = facet_value %Child, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.606: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9ab) [concrete]
// CHECK:STDOUT:   %facet_value.5af: %facet_type = facet_value %Child, (%DestroyImpl.impl_witness.606, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.af6: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.5af) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base = import_ref Main//a, Base, unloaded
// CHECK:STDOUT:   %Main.Child: type = import_ref Main//a, Child, loaded [concrete = constants.%Child]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.739: <witness> = import_ref Main//a, loc10_1, loaded [concrete = constants.%complete_type.d98]
// CHECK:STDOUT:   %Main.import_ref.d68 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bc9: %Base.F.type = import_ref Main//a, loc5_13, loaded [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Main.import_ref.077 = import_ref Main//a, loc6_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.25c: %Base.elem = import_ref Main//a, loc8_8, loaded [concrete = %field_decl.c32]
// CHECK:STDOUT:   %Main.import_ref.029: %Base.elem = import_ref Main//a, loc9_15, loaded [concrete = %field_decl.80a]
// CHECK:STDOUT:   %Main.import_ref.439419.1: type = import_ref Main//a, loc13_16, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %.754: %Child.elem = base_decl %Main.import_ref.439419.1, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.60e: <witness> = import_ref Main//a, loc14_1, loaded [concrete = constants.%complete_type.c02]
// CHECK:STDOUT:   %Main.import_ref.6e8 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d51: %Child.elem = import_ref Main//a, loc13_20, loaded [concrete = %.754]
// CHECK:STDOUT:   %Main.import_ref.439419.2: type = import_ref Main//a, loc13_16, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.dc5ddb.1 = import_ref Main//a, loc8_10, unloaded
// CHECK:STDOUT:   %field_decl.c32: %Base.elem = field_decl x, element0, %Main.import_ref.dc5ddb.1 in [concrete] {}
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.688: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.688), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.dc5ddb.2 = import_ref Main//a, loc9_17, unloaded
// CHECK:STDOUT:   %field_decl.80a: %Base.elem = field_decl unused_y, element1, %Main.import_ref.dc5ddb.2 in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = imports.%Main.Base
// CHECK:STDOUT:     .Child = imports.%Main.Child
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.60e
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6e8
// CHECK:STDOUT:   .base = imports.%Main.import_ref.d51
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.439419.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.739
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d68
// CHECK:STDOUT:   .F = imports.%Main.import_ref.bc9
// CHECK:STDOUT:   .Unused = imports.%Main.import_ref.077
// CHECK:STDOUT:   .x = imports.%Main.import_ref.25c
// CHECK:STDOUT:   .unused_y = imports.%Main.import_ref.029
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Child = var_storage %a.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_49.1: %struct_type.x.unused_y.76a = struct_literal (%int_0, %int_1) [concrete = constants.%struct.0bf]
// CHECK:STDOUT:   %.loc7_50.1: %struct_type.base.503 = struct_literal (%.loc7_49.1) [concrete = constants.%struct.cb5]
// CHECK:STDOUT:   %impl.elem0.loc7_49.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc7_49.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da]
// CHECK:STDOUT:   %specific_fn.loc7_49.1: <specific function> = specific_function %impl.elem0.loc7_49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.2: <bound method> = bound_method %int_0, %specific_fn.loc7_49.1 [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1: init %i32 = call %bound_method.loc7_49.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc7_49.2: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc7_50.2: ref %.edf = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_49.3: ref %i32 = class_element_access %.loc7_50.2, element0
// CHECK:STDOUT:   %.loc7_49.4: init %i32 to %.loc7_49.3 = in_place_init %.loc7_49.2 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %impl.elem0.loc7_49.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc7_49.3: <bound method> = bound_method %int_1, %impl.elem0.loc7_49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc7_49.2: <specific function> = specific_function %impl.elem0.loc7_49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_49.4: <bound method> = bound_method %int_1, %specific_fn.loc7_49.2 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2: init %i32 = call %bound_method.loc7_49.4(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc7_49.5: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_49.2 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc7_49.6: ref %i32 = class_element_access %.loc7_50.2, element1
// CHECK:STDOUT:   %.loc7_49.7: init %i32 to %.loc7_49.6 = in_place_init %.loc7_49.5 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc7_49.8: init %.edf to %.loc7_50.2 = class_init (%.loc7_49.4, %.loc7_49.7) [concrete = constants.%struct.bd4]
// CHECK:STDOUT:   %.loc7_50.3: init %.edf = converted %.loc7_49.1, %.loc7_49.8 [concrete = constants.%struct.bd4]
// CHECK:STDOUT:   %.loc7_50.4: init %Base = as_compatible %.loc7_50.3 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_50.5: init %Child to %a.var = class_init (%.loc7_50.4) [concrete = constants.%Child.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Child = converted %.loc7_50.1, %.loc7_50.5 [concrete = constants.%Child.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, imports.%Main.Child [concrete = constants.%Child]
// CHECK:STDOUT:   %a: ref %Child = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.860 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.860 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%Main.import_ref.25c [concrete = imports.%field_decl.c32]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc8_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_7.2: <bound method> = bound_method %int_2, %specific_fn.loc8 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_7.2(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc8_7: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, imports.%Main.import_ref.bc9 [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_3.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.2: ref %Base = converted %a.ref.loc9, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.3: %Base = acquire_value %.loc9_3.2
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %Base.F.bound(%.loc9_3.3)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Child, (constants.%DestroyImpl.impl_witness.606, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.5af]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Child, %facet_value.loc7_3.1 [concrete = constants.%facet_value.5af]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Child, (constants.%DestroyImpl.impl_witness.606, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.5af]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Child, %facet_value.loc7_3.2 [concrete = constants.%facet_value.5af]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.5af) [concrete = constants.%SelfDestruct.specific_fn.af6]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %struct_type.x.unused_y.20e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_3.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%i32, %facet_value.loc7_3.1 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%i32, %facet_value.loc7_3.2 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.c23) [concrete = constants.%SelfDestruct.specific_fn.da9]
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_3.1)
// CHECK:STDOUT:   %.loc7_3.4: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc7_3.3: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.5: %facet_type = converted constants.%i32, %facet_value.loc7_3.3 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %facet_value.loc7_3.4: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.6: %facet_type = converted constants.%i32, %facet_value.loc7_3.4 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.c23) [concrete = constants.%SelfDestruct.specific_fn.da9]
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc7_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.3(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %unused_y.ref: %Base.elem = name_ref unused_y, imports.%Main.import_ref.029 [concrete = imports.%field_decl.80a]
// CHECK:STDOUT:   %.loc7_3.1: ref %i32 = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%i32, %facet_value.loc7_3.1 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%i32, %facet_value.loc7_3.2 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.c23) [concrete = constants.%SelfDestruct.specific_fn.da9]
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_3.1)
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%Main.import_ref.25c [concrete = imports.%field_decl.c32]
// CHECK:STDOUT:   %.loc7_3.4: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc7_3.3: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.5: %facet_type = converted constants.%i32, %facet_value.loc7_3.3 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %facet_value.loc7_3.4: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.52c, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %.loc7_3.6: %facet_type = converted constants.%i32, %facet_value.loc7_3.4 [concrete = constants.%facet_value.c23]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.c23) [concrete = constants.%SelfDestruct.specific_fn.da9]
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc7_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.4(%self.param: ref %struct_type.base.cae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_3.1: ref %Base = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Base, (constants.%DestroyImpl.impl_witness.69a, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Base, %facet_value.loc7_3.1 [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Base, (constants.%DestroyImpl.impl_witness.69a, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Base, %facet_value.loc7_3.2 [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.0ce) [concrete = constants.%SelfDestruct.specific_fn.de2]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc7_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.5(%self.param: ref %Child) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Child.elem = name_ref base, imports.%Main.import_ref.d51 [concrete = imports.%.754]
// CHECK:STDOUT:   %.loc7_3.1: ref %Base = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Base, (constants.%DestroyImpl.impl_witness.69a, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Base, %facet_value.loc7_3.1 [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Base, (constants.%DestroyImpl.impl_witness.69a, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Base, %facet_value.loc7_3.2 [concrete = constants.%facet_value.0ce]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.0ce) [concrete = constants.%SelfDestruct.specific_fn.de2]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc7_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
