// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface Empty {
}

interface Basic {
  let T:! type;
  fn F();
}

// TODO: Implement merging of interface on import.
// interface ForwardDeclared;

interface ForwardDeclared {
  let T:! type;
  fn F();
}

var f_ref: {.f: ForwardDeclared};

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn UseEmpty(unused e: Empty) {}
fn UseBasic(unused e: Basic) {}
fn UseForwardDeclared(unused f: ForwardDeclared) {}

alias UseBasicT = Basic.T;
alias UseBasicF = Basic.F;

alias UseForwardDeclaredT = ForwardDeclared.T;
alias UseForwardDeclaredF = ForwardDeclared.F;

var f: ForwardDeclared* = &f_ref.f;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.be5: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.a88: %Basic.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.1ec: %Basic.assoc_type = assoc_entity element0, @Basic.WithSelf.%T [concrete]
// CHECK:STDOUT:   %Basic.WithSelf.F.type: type = fn_type @Basic.WithSelf.F, @Basic.WithSelf(%Self.a88) [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F: %Basic.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.eed: %Basic.assoc_type = assoc_entity element1, @Basic.WithSelf.%Basic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.cf4: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.ebe: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.WithSelf.%T [concrete]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type: type = fn_type @ForwardDeclared.WithSelf.F, @ForwardDeclared.WithSelf(%Self.cf4) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F: %ForwardDeclared.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.c88: %ForwardDeclared.assoc_type = assoc_entity element1, @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type.ff2: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.d5a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.9f4: %T.as.DefaultOrUnformed.impl.Op.type.d5a = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.e73: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f82, @T.as.DefaultOrUnformed.impl(%struct_type.f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.aae: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%struct_type.f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.1a4: %T.as.DefaultOrUnformed.impl.Op.type.aae = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %struct_type.f, (%DefaultOrUnformed.impl_witness.e73) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.1a4, @T.as.DefaultOrUnformed.impl.Op(%struct_type.f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.b30: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.d5a) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.9f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f82 = impl_witness_table (%Core.import_ref.b30), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Basic = %Basic.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl
// CHECK:STDOUT:     .f_ref = %f_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %Basic.decl: type = interface_decl @Basic [concrete = constants.%Basic.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl: type = interface_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f_ref.patt: %pattern_type.ff2 = ref_binding_pattern f_ref [concrete]
// CHECK:STDOUT:     %f_ref.var_patt: %pattern_type.ff2 = var_pattern %f_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f_ref.var: ref %struct_type.f = var %f_ref.var_patt [concrete]
// CHECK:STDOUT:   %.loc20: type = splice_block %struct_type.f [concrete = constants.%struct_type.f] {
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f_ref: ref %struct_type.f = ref_binding f_ref, %f_ref.var [concrete = %f_ref.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.be5]
// CHECK:STDOUT:   %Empty.WithSelf.decl = interface_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %Self: %Basic.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a88]
// CHECK:STDOUT:   %Basic.WithSelf.decl = interface_with_self_decl @Basic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc8 [concrete] {
// CHECK:STDOUT:     %assoc0: %Basic.assoc_type = assoc_entity element0, @Basic.WithSelf.%T [concrete = constants.%assoc0.1ec]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.WithSelf.F.decl: @Basic.WithSelf.%Basic.WithSelf.F.type (%Basic.WithSelf.F.type) = fn_decl @Basic.WithSelf.F [symbolic = @Basic.WithSelf.%Basic.WithSelf.F (constants.%Basic.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc1: %Basic.assoc_type = assoc_entity element1, %Basic.WithSelf.F.decl [concrete = constants.%assoc1.eed]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc8.%assoc0
// CHECK:STDOUT:   .F = @Basic.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@Basic.WithSelf.%T, @Basic.WithSelf.%Basic.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cf4]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.decl = interface_with_self_decl @ForwardDeclared [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc16 [concrete] {
// CHECK:STDOUT:     %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.WithSelf.%T [concrete = constants.%assoc0.ebe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.decl: @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.type (%ForwardDeclared.WithSelf.F.type) = fn_decl @ForwardDeclared.WithSelf.F [symbolic = @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F (constants.%ForwardDeclared.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc1: %ForwardDeclared.assoc_type = assoc_entity element1, %ForwardDeclared.WithSelf.F.decl [concrete = constants.%assoc1.c88]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc16.%assoc0
// CHECK:STDOUT:   .F = @ForwardDeclared.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@ForwardDeclared.WithSelf.%T, @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Basic.WithSelf.F(@Basic.%Self: %Basic.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared.WithSelf.F(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%struct_type.f, (constants.%DefaultOrUnformed.impl_witness.e73) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc20_33.1: %DefaultOrUnformed.type = converted constants.%struct_type.f, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc20_33.1 [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   %.loc20_33.2: type = converted %.loc20_33.1, %as_type [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.1a4, @T.as.DefaultOrUnformed.impl.Op(constants.%struct_type.f) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %struct_type.f = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%f_ref.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.be5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf(constants.%Self.a88) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf.F(constants.%Self.a88) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf(constants.%Self.cf4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.cf4
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type => constants.%ForwardDeclared.WithSelf.F.type
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F => constants.%ForwardDeclared.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf.F(constants.%Self.cf4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.688: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a11: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [concrete]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.51d: %Basic.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F.type: type = fn_type @Basic.WithSelf.F, @Basic.WithSelf(%Self.51d) [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F: %Basic.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.aa9: type = pattern_type %Basic.type [concrete]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [concrete]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.656: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type: type = fn_type @ForwardDeclared.WithSelf.F, @ForwardDeclared.WithSelf(%Self.656) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F: %ForwardDeclared.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.490: type = pattern_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [concrete]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.cdf: %Basic.assoc_type = assoc_entity element0, imports.%Main.import_ref.633 [concrete]
// CHECK:STDOUT:   %assoc1.cd7: %Basic.assoc_type = assoc_entity element1, imports.%Main.import_ref.05f [concrete]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.ded: %ForwardDeclared.assoc_type = assoc_entity element0, imports.%Main.import_ref.fe3 [concrete]
// CHECK:STDOUT:   %assoc1.d3c: %ForwardDeclared.assoc_type = assoc_entity element1, imports.%Main.import_ref.77c [concrete]
// CHECK:STDOUT:   %ptr.138: type = ptr_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type %ptr.138 [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type.7ad: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT:   %.0e2: ref %ForwardDeclared.type = struct_access imports.%f_ref.var, element0 [concrete]
// CHECK:STDOUT:   %addr: %ptr.138 = addr_of %.0e2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.cb4: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.124: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.4c1: %ptr.as.Copy.impl.Op.type.124 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.138, (%Copy.impl_witness.cb4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3ba: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3d9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3ba, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.4c1 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.4c1, @ptr.as.Copy.impl.Op(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Empty: type = import_ref Main//a, Empty, loaded [concrete = constants.%Empty.type]
// CHECK:STDOUT:   %Main.Basic: type = import_ref Main//a, Basic, loaded [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %Main.ForwardDeclared: type = import_ref Main//a, ForwardDeclared, loaded [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %Main.f_ref: ref %struct_type.f = import_ref Main//a, f_ref, loaded [concrete = %f_ref.var]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8dd = import_ref Main//a, loc4_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.8d1: %Basic.assoc_type = import_ref Main//a, loc8_8, loaded [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:   %Main.import_ref.30b: %Basic.assoc_type = import_ref Main//a, loc9_9, loaded [concrete = constants.%assoc1.cd7]
// CHECK:STDOUT:   %Main.T.0c0 = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.35f = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.8e6a30.2: %Basic.type = import_ref Main//a, loc7_17, loaded [symbolic = constants.%Self.51d]
// CHECK:STDOUT:   %Main.import_ref.2e7 = import_ref Main//a, loc7_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.d14: %ForwardDeclared.assoc_type = import_ref Main//a, loc16_8, loaded [concrete = constants.%assoc0.ded]
// CHECK:STDOUT:   %Main.import_ref.433: %ForwardDeclared.assoc_type = import_ref Main//a, loc17_9, loaded [concrete = constants.%assoc1.d3c]
// CHECK:STDOUT:   %Main.T.7c6 = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.dba = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.56cd0d.2: %ForwardDeclared.type = import_ref Main//a, loc15_27, loaded [symbolic = constants.%Self.656]
// CHECK:STDOUT:   %Main.import_ref.e52 = import_ref Main//a, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.633 = import_ref Main//a, loc8_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.05f = import_ref Main//a, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.fe3 = import_ref Main//a, loc16_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.77c = import_ref Main//a, loc17_9, unloaded
// CHECK:STDOUT:   %f_ref.patt: %pattern_type.7ad = ref_binding_pattern f_ref [concrete]
// CHECK:STDOUT:   %f_ref.var_patt: %pattern_type.7ad = var_pattern %f_ref.patt [concrete]
// CHECK:STDOUT:   %f_ref.var: ref %struct_type.f = var %f_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = imports.%Main.Empty
// CHECK:STDOUT:     .Basic = imports.%Main.Basic
// CHECK:STDOUT:     .ForwardDeclared = imports.%Main.ForwardDeclared
// CHECK:STDOUT:     .f_ref = imports.%Main.f_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UseEmpty = %UseEmpty.decl
// CHECK:STDOUT:     .UseBasic = %UseBasic.decl
// CHECK:STDOUT:     .UseForwardDeclared = %UseForwardDeclared.decl
// CHECK:STDOUT:     .UseBasicT = %UseBasicT
// CHECK:STDOUT:     .UseBasicF = %UseBasicF
// CHECK:STDOUT:     .UseForwardDeclaredT = %UseForwardDeclaredT
// CHECK:STDOUT:     .UseForwardDeclaredF = %UseForwardDeclaredF
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %UseEmpty.decl: %UseEmpty.type = fn_decl @UseEmpty [concrete = constants.%UseEmpty] {
// CHECK:STDOUT:     %e.param_patt: %pattern_type.a11 = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.a11 = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Empty.type = value_param call_param0
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty.type]
// CHECK:STDOUT:     %e: %Empty.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [concrete = constants.%UseBasic] {
// CHECK:STDOUT:     %e.param_patt: %pattern_type.aa9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.aa9 = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Basic.type = value_param call_param0
// CHECK:STDOUT:     %Basic.ref: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:     %e: %Basic.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [concrete = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %f.param_patt: %pattern_type.490 = value_param_pattern [concrete]
// CHECK:STDOUT:     %f.patt: %pattern_type.490 = at_binding_pattern f, %f.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %f.param: %ForwardDeclared.type = value_param call_param0
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %f: %ForwardDeclared.type = value_binding f, %f.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %T.ref.loc10: %Basic.assoc_type = name_ref T, imports.%Main.import_ref.8d1 [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:   %UseBasicT: %Basic.assoc_type = alias_binding UseBasicT, %T.ref.loc10 [concrete = constants.%assoc0.cdf]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %F.ref.loc11: %Basic.assoc_type = name_ref F, imports.%Main.import_ref.30b [concrete = constants.%assoc1.cd7]
// CHECK:STDOUT:   %UseBasicF: %Basic.assoc_type = alias_binding UseBasicF, %F.ref.loc11 [concrete = constants.%assoc1.cd7]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %T.ref.loc13: %ForwardDeclared.assoc_type = name_ref T, imports.%Main.import_ref.d14 [concrete = constants.%assoc0.ded]
// CHECK:STDOUT:   %UseForwardDeclaredT: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredT, %T.ref.loc13 [concrete = constants.%assoc0.ded]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %F.ref.loc14: %ForwardDeclared.assoc_type = name_ref F, imports.%Main.import_ref.433 [concrete = constants.%assoc1.d3c]
// CHECK:STDOUT:   %UseForwardDeclaredF: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredF, %F.ref.loc14 [concrete = constants.%assoc1.d3c]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.13f = ref_binding_pattern f [concrete]
// CHECK:STDOUT:     %f.var_patt: %pattern_type.13f = var_pattern %f.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %ptr.138 = var %f.var_patt [concrete]
// CHECK:STDOUT:   %.loc16: type = splice_block %ptr [concrete = constants.%ptr.138] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %ptr: type = ptr_type %ForwardDeclared.ref.loc16 [concrete = constants.%ptr.138]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %ptr.138 = ref_binding f, %f.var [concrete = %f.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8dd
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2e7
// CHECK:STDOUT:   .T = imports.%Main.import_ref.8d1
// CHECK:STDOUT:   .F = imports.%Main.import_ref.30b
// CHECK:STDOUT:   witness = (imports.%Main.T.0c0, imports.%Main.F.35f)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e52
// CHECK:STDOUT:   .T = imports.%Main.import_ref.d14
// CHECK:STDOUT:   .F = imports.%Main.import_ref.433
// CHECK:STDOUT:   witness = (imports.%Main.T.7c6, imports.%Main.F.dba)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e.param: %Empty.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Basic.WithSelf.F(imports.%Main.import_ref.8e6a30.2: %Basic.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e.param: %Basic.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared.WithSelf.F(imports.%Main.import_ref.56cd0d.2: %ForwardDeclared.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f.param: %ForwardDeclared.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref %struct_type.f = name_ref f_ref, imports.%Main.f_ref [concrete = imports.%f_ref.var]
// CHECK:STDOUT:   %.loc16: ref %ForwardDeclared.type = struct_access %f_ref.ref, element0 [concrete = constants.%.0e2]
// CHECK:STDOUT:   %addr: %ptr.138 = addr_of %.loc16 [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.3d9 = impl_witness_access constants.%Copy.impl_witness.cb4, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4c1]
// CHECK:STDOUT:   %bound_method.loc16_27.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ForwardDeclared.type) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_27.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.138 = call %bound_method.loc16_27.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%f.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.688) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf(constants.%Self.51d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.51d
// CHECK:STDOUT:   %Basic.WithSelf.F.type => constants.%Basic.WithSelf.F.type
// CHECK:STDOUT:   %Basic.WithSelf.F => constants.%Basic.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf.F(constants.%Self.51d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf(constants.%Self.656) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type => constants.%ForwardDeclared.WithSelf.F.type
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F => constants.%ForwardDeclared.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf.F(constants.%Self.656) {}
// CHECK:STDOUT:
