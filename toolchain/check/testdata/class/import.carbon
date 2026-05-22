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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Empty {
}

class Field {
  var x: i32;
}

class ForwardDeclared;

class ForwardDeclared {
  fn F[self: Self]();
  fn G[ref self: Self]();
}

class Incomplete;

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var unused a: Empty = {};

  var b: Field = {.x = 1};
  b.x = 2;

  var c: ForwardDeclared = {};
  c.F();
  c.G();

  var unused d: ForwardDeclared* = &c;

  var unused e: Incomplete*;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Field: type = class_type @Field [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Field.elem: type = unbound_element_type %Field, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %ForwardDeclared: type = class_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %pattern_type.d21: type = pattern_type %ForwardDeclared [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F.type: type = fn_type @ForwardDeclared.F [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F: %ForwardDeclared.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.G.type: type = fn_type @ForwardDeclared.G [concrete]
// CHECK:STDOUT:   %ForwardDeclared.G: %ForwardDeclared.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
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
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Field = %Field.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc11
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = class_decl @Empty [concrete = constants.%Empty] {} {}
// CHECK:STDOUT:   %Field.decl: type = class_decl @Field [concrete = constants.%Field] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc11: type = class_decl @ForwardDeclared [concrete = constants.%ForwardDeclared] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc13: type = class_decl @ForwardDeclared [concrete = constants.%ForwardDeclared] {} {}
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Empty
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field {
// CHECK:STDOUT:   %.loc8: %Field.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.fa5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Field
// CHECK:STDOUT:   .x = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared {
// CHECK:STDOUT:   %ForwardDeclared.F.decl: %ForwardDeclared.F.type = fn_decl @ForwardDeclared.F [concrete = constants.%ForwardDeclared.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d21 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.d21 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ForwardDeclared = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDeclared [concrete = constants.%ForwardDeclared]
// CHECK:STDOUT:     %self: %ForwardDeclared = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclared.G.decl: %ForwardDeclared.G.type = fn_decl @ForwardDeclared.G [concrete = constants.%ForwardDeclared.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d21 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.d21 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ForwardDeclared = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDeclared [concrete = constants.%ForwardDeclared]
// CHECK:STDOUT:     %self: ref %ForwardDeclared = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardDeclared
// CHECK:STDOUT:   .F = %ForwardDeclared.F.decl
// CHECK:STDOUT:   .G = %ForwardDeclared.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardDeclared.F(%self.param: %ForwardDeclared);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardDeclared.G(%self.param: ref %ForwardDeclared);
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty: type = class_type @Empty [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.4b2: type = pattern_type %Empty [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.val: %Empty = struct_value () [concrete]
// CHECK:STDOUT:   %Field: type = class_type @Field [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %struct_type.x.740: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.9ec: <witness> = complete_type_witness %struct_type.x.740 [concrete]
// CHECK:STDOUT:   %pattern_type.272: type = pattern_type %Field [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.c96 = struct_value (%int_1.5b8) [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.924 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.924, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b34: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.50a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Field.val: %Field = struct_value (%int_1.50a) [concrete]
// CHECK:STDOUT:   %Field.elem: type = unbound_element_type %Field, %i32 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f4c: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.924 [concrete]
// CHECK:STDOUT:   %bound_method.fe7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.21f: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %ForwardDeclared.63cd1a.1: type = class_type @ForwardDeclared.1 [concrete]
// CHECK:STDOUT:   %pattern_type.6805: type = pattern_type %ForwardDeclared.63cd1a.1 [concrete]
// CHECK:STDOUT:   %ForwardDeclared.val: %ForwardDeclared.63cd1a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F.type: type = fn_type @ForwardDeclared.F [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F: %ForwardDeclared.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.G.type: type = fn_type @ForwardDeclared.G [concrete]
// CHECK:STDOUT:   %ForwardDeclared.G: %ForwardDeclared.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3ab: type = ptr_type %ForwardDeclared.63cd1a.1 [concrete]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %ptr.3ab [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f9b: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%ForwardDeclared.63cd1a.1) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.bdc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ForwardDeclared.63cd1a.1) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.322: %ptr.as.Copy.impl.Op.type.bdc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.3ab, (%Copy.impl_witness.f9b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5b6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.f75: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5b6, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.322, @ptr.as.Copy.impl.Op(%ForwardDeclared.63cd1a.1) [concrete]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %ptr.c81: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT:   %pattern_type.9d7: type = pattern_type %ptr.c81 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.d5a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.9f4: %T.as.DefaultOrUnformed.impl.Op.type.d5a = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.6ee: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f82, @T.as.DefaultOrUnformed.impl(%ptr.c81) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.ac9: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%ptr.c81) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.012: %T.as.DefaultOrUnformed.impl.Op.type.ac9 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %ptr.c81, (%DefaultOrUnformed.impl_witness.6ee) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.012, @T.as.DefaultOrUnformed.impl.Op(%ptr.c81) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.8: type = fn_type @Destroy.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.8: %Destroy.Op.type.af7ec0.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.9: type = fn_type @Destroy.Op.loc7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.9: %Destroy.Op.type.af7ec0.9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Empty: type = import_ref Main//a, Empty, loaded [concrete = constants.%Empty]
// CHECK:STDOUT:   %Main.Field: type = import_ref Main//a, Field, loaded [concrete = constants.%Field]
// CHECK:STDOUT:   %Main.ForwardDeclared: type = import_ref Main//a, ForwardDeclared, loaded [concrete = constants.%ForwardDeclared.63cd1a.1]
// CHECK:STDOUT:   %Main.Incomplete: type = import_ref Main//a, Incomplete, loaded [concrete = constants.%Incomplete]
// CHECK:STDOUT:   %Core.b38: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//a, loc5_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.121 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bc4: <witness> = import_ref Main//a, loc9_1, loaded [concrete = constants.%complete_type.9ec]
// CHECK:STDOUT:   %Main.import_ref.475 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.3e4: %Field.elem = import_ref Main//a, loc8_8, loaded [concrete = %.3b6]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7f1: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ca9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.768)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bc3 = impl_witness_table (%Core.import_ref.7f1), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %.3b6: %Field.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//a, loc16_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.1dc190.1 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e20: %ForwardDeclared.F.type = import_ref Main//a, loc14_21, loaded [concrete = constants.%ForwardDeclared.F]
// CHECK:STDOUT:   %Main.import_ref.43c: %ForwardDeclared.G.type = import_ref Main//a, loc15_25, loaded [concrete = constants.%ForwardDeclared.G]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//a, loc16_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.1dc190.2 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.aee = import_ref Main//a, loc14_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.5d4 = import_ref Main//a, loc15_25, unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.b30: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.d5a) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.9f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f82 = impl_witness_table (%Core.import_ref.b30), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = imports.%Main.Empty
// CHECK:STDOUT:     .Field = imports.%Main.Field
// CHECK:STDOUT:     .ForwardDeclared = imports.%Main.ForwardDeclared
// CHECK:STDOUT:     .Incomplete = imports.%Main.Incomplete
// CHECK:STDOUT:     .Core = imports.%Core.b38
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.bc4
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.475
// CHECK:STDOUT:   .x = imports.%Main.import_ref.3e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.1 [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1dc190.1
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e20
// CHECK:STDOUT:   .G = imports.%Main.import_ref.43c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.2 [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1dc190.2
// CHECK:STDOUT:   .F = imports.%Main.import_ref.aee
// CHECK:STDOUT:   .G = imports.%Main.import_ref.5d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.4b2 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.4b2 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Empty = var %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Empty to %a.var = class_init () [concrete = constants.%Empty.val]
// CHECK:STDOUT:   %.loc7_3: init %Empty = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Empty.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3
// CHECK:STDOUT:   %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty]
// CHECK:STDOUT:   %a: ref %Empty = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.272 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.272 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Field = var %b.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_25.1: %struct_type.x.c96 = struct_literal (%int_1) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc9: %.634 = impl_witness_access constants.%ImplicitAs.impl_witness.1e6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.924]
// CHECK:STDOUT:   %bound_method.loc9_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff4]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_25.2: <bound method> = bound_method %int_1, %specific_fn.loc9 [concrete = constants.%bound_method.b34]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_25.2(%int_1) [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc9_25.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc9_25.3: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc9_25.4: init %i32 to %.loc9_25.3 = in_place_init %.loc9_25.2 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:   %.loc9_25.5: init %Field to %b.var = class_init (%.loc9_25.4) [concrete = constants.%Field.val]
// CHECK:STDOUT:   %.loc9_3: init %Field = converted %.loc9_25.1, %.loc9_25.5 [concrete = constants.%Field.val]
// CHECK:STDOUT:   assign %b.var, %.loc9_3
// CHECK:STDOUT:   %Field.ref: type = name_ref Field, imports.%Main.Field [concrete = constants.%Field]
// CHECK:STDOUT:   %b: ref %Field = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref: ref %Field = name_ref b, %b
// CHECK:STDOUT:   %x.ref: %Field.elem = name_ref x, imports.%Main.import_ref.3e4 [concrete = imports.%.3b6]
// CHECK:STDOUT:   %.loc10_4: ref %i32 = class_element_access %b.ref, element0
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc10: %.634 = impl_witness_access constants.%ImplicitAs.impl_witness.1e6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.924]
// CHECK:STDOUT:   %bound_method.loc10_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f4c]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_7.2: <bound method> = bound_method %int_2, %specific_fn.loc10 [concrete = constants.%bound_method.fe7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_7.2(%int_2) [concrete = constants.%int_2.21f]
// CHECK:STDOUT:   %.loc10_7: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_2.21f]
// CHECK:STDOUT:   assign %.loc10_4, %.loc10_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6805 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.6805 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %ForwardDeclared.63cd1a.1 = var %c.var_patt
// CHECK:STDOUT:   %.loc12_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_29.2: init %ForwardDeclared.63cd1a.1 to %c.var = class_init () [concrete = constants.%ForwardDeclared.val]
// CHECK:STDOUT:   %.loc12_3: init %ForwardDeclared.63cd1a.1 = converted %.loc12_29.1, %.loc12_29.2 [concrete = constants.%ForwardDeclared.val]
// CHECK:STDOUT:   assign %c.var, %.loc12_3
// CHECK:STDOUT:   %ForwardDeclared.ref.loc12: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.63cd1a.1]
// CHECK:STDOUT:   %c: ref %ForwardDeclared.63cd1a.1 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc13: ref %ForwardDeclared.63cd1a.1 = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %ForwardDeclared.F.type = name_ref F, imports.%Main.import_ref.e20 [concrete = constants.%ForwardDeclared.F]
// CHECK:STDOUT:   %ForwardDeclared.F.bound: <bound method> = bound_method %c.ref.loc13, %F.ref
// CHECK:STDOUT:   %.loc13: %ForwardDeclared.63cd1a.1 = acquire_value %c.ref.loc13
// CHECK:STDOUT:   %ForwardDeclared.F.call: init %empty_tuple.type = call %ForwardDeclared.F.bound(%.loc13)
// CHECK:STDOUT:   %c.ref.loc14: ref %ForwardDeclared.63cd1a.1 = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %ForwardDeclared.G.type = name_ref G, imports.%Main.import_ref.43c [concrete = constants.%ForwardDeclared.G]
// CHECK:STDOUT:   %ForwardDeclared.G.bound: <bound method> = bound_method %c.ref.loc14, %G.ref
// CHECK:STDOUT:   %ForwardDeclared.G.call: init %empty_tuple.type = call %ForwardDeclared.G.bound(%c.ref.loc14)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.14b = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.14b = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %ptr.3ab = var %d.var_patt
// CHECK:STDOUT:   %c.ref.loc16: ref %ForwardDeclared.63cd1a.1 = name_ref c, %c
// CHECK:STDOUT:   %addr: %ptr.3ab = addr_of %c.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16: %.f75 = impl_witness_access constants.%Copy.impl_witness.f9b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.322]
// CHECK:STDOUT:   %bound_method.loc16_36.1: <bound method> = bound_method %addr, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @ptr.as.Copy.impl.Op(constants.%ForwardDeclared.63cd1a.1) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_36.2: <bound method> = bound_method %addr, %specific_fn.loc16
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.3ab = call %bound_method.loc16_36.2(%addr)
// CHECK:STDOUT:   assign %d.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc16: type = splice_block %ptr.loc16 [concrete = constants.%ptr.3ab] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.63cd1a.1]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %ForwardDeclared.ref.loc16 [concrete = constants.%ptr.3ab]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %ptr.3ab = ref_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.9d7 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.9d7 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %ptr.c81 = var %e.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%ptr.c81, (constants.%DefaultOrUnformed.impl_witness.6ee) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc18_28.1: %DefaultOrUnformed.type = converted constants.%ptr.c81, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc18_28.1 [concrete = constants.%ptr.c81]
// CHECK:STDOUT:   %.loc18_28.2: type = converted %.loc18_28.1, %as_type [concrete = constants.%ptr.c81]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.012, @T.as.DefaultOrUnformed.impl.Op(constants.%ptr.c81) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %ptr.c81 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %e.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc18_27: type = splice_block %ptr.loc18 [concrete = constants.%ptr.c81] {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, imports.%Main.Incomplete [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %Incomplete.ref [concrete = constants.%ptr.c81]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %ptr.c81 = ref_binding e, %e.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %e.var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%e.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%d.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %c.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.1dc86d.8
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc7: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.9
// CHECK:STDOUT:   %Destroy.Op.call.loc7: init %empty_tuple.type = call %Destroy.Op.bound.loc7(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardDeclared.F [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardDeclared.G [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18(%self.param: ref %ptr.c81) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16(%self.param: ref %ptr.3ab) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.2(%self.param: ref %ForwardDeclared.63cd1a.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %struct_type.x.740) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.4(%self.param: ref %Field) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7(%self.param: ref %Empty) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
