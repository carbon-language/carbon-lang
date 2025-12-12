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

fn UseEmpty(e: Empty) {}
fn UseBasic(e: Basic) {}
fn UseForwardDeclared(f: ForwardDeclared) {}

alias UseBasicT = Basic.T;
alias UseBasicF = Basic.F;

alias UseForwardDeclaredT = ForwardDeclared.T;
alias UseForwardDeclaredF = ForwardDeclared.F;

var f: ForwardDeclared* = &f_ref.f;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.61e: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.3e6: %Basic.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.909: %Basic.assoc_type = assoc_entity element0, @Basic.%T [concrete]
// CHECK:STDOUT:   %Basic.F.type: type = fn_type @Basic.F [concrete]
// CHECK:STDOUT:   %Basic.F: %Basic.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.a1d: %Basic.assoc_type = assoc_entity element1, @Basic.%Basic.F.decl [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.8e7: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.06c: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.%T [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F.type: type = fn_type @ForwardDeclared.F [concrete]
// CHECK:STDOUT:   %ForwardDeclared.F: %ForwardDeclared.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.b1d: %ForwardDeclared.assoc_type = assoc_entity element1, @ForwardDeclared.%ForwardDeclared.F.decl [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %f_ref.patt: %pattern_type = ref_binding_pattern f_ref [concrete]
// CHECK:STDOUT:     %f_ref.var_patt: %pattern_type = var_pattern %f_ref.patt [concrete]
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
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.61e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %Self: %Basic.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3e6]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc8 [concrete] {
// CHECK:STDOUT:     %assoc0: %Basic.assoc_type = assoc_entity element0, @Basic.%T [concrete = constants.%assoc0.909]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.F.decl: %Basic.F.type = fn_decl @Basic.F [concrete = constants.%Basic.F] {} {}
// CHECK:STDOUT:   %assoc1: %Basic.assoc_type = assoc_entity element1, %Basic.F.decl [concrete = constants.%assoc1.a1d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc8.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %Basic.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8e7]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc16 [concrete] {
// CHECK:STDOUT:     %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.%T [concrete = constants.%assoc0.06c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclared.F.decl: %ForwardDeclared.F.type = fn_decl @ForwardDeclared.F [concrete = constants.%ForwardDeclared.F] {} {}
// CHECK:STDOUT:   %assoc1: %ForwardDeclared.assoc_type = assoc_entity element1, %ForwardDeclared.F.decl [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc16.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %ForwardDeclared.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T.loc8(@Basic.%Self: %Basic.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T.loc16(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Basic.F(@Basic.%Self: %Basic.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared.F(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.loc8(constants.%Self.3e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.F(constants.%Self.3e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.loc16(constants.%Self.8e7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.F(constants.%Self.8e7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %pattern_type.7d1: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [concrete]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %pattern_type.185: type = pattern_type %Basic.type [concrete]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [concrete]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %pattern_type.0a0: type = pattern_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [concrete]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.f14: %Basic.assoc_type = assoc_entity element0, imports.%Main.import_ref.a37 [concrete]
// CHECK:STDOUT:   %assoc1.8cf: %Basic.assoc_type = assoc_entity element1, imports.%Main.import_ref.e5c [concrete]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.d49: %ForwardDeclared.assoc_type = assoc_entity element0, imports.%Main.import_ref.e0c [concrete]
// CHECK:STDOUT:   %assoc1.128: %ForwardDeclared.assoc_type = assoc_entity element1, imports.%Main.import_ref.b1c [concrete]
// CHECK:STDOUT:   %ptr.c31: type = ptr_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %pattern_type.063: type = pattern_type %ptr.c31 [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT:   %.816: ref %ForwardDeclared.type = struct_access imports.%f_ref.var, element0 [concrete]
// CHECK:STDOUT:   %addr: %ptr.c31 = addr_of %.816 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e5d: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ba9: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.a2e: %ptr.as.Copy.impl.Op.type.ba9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.c31, (%Copy.impl_witness.e5d) [concrete]
// CHECK:STDOUT:   %.5f3: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.a2e [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.a2e, @ptr.as.Copy.impl.Op(%ForwardDeclared.type) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.69f = import_ref Main//a, loc4_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.1e4 = import_ref Main//a, loc7_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.016: %Basic.assoc_type = import_ref Main//a, loc8_8, loaded [concrete = constants.%assoc0.f14]
// CHECK:STDOUT:   %Main.import_ref.f57: %Basic.assoc_type = import_ref Main//a, loc9_9, loaded [concrete = constants.%assoc1.8cf]
// CHECK:STDOUT:   %Main.T.faa = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.dff = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.952 = import_ref Main//a, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.d48: %ForwardDeclared.assoc_type = import_ref Main//a, loc16_8, loaded [concrete = constants.%assoc0.d49]
// CHECK:STDOUT:   %Main.import_ref.176: %ForwardDeclared.assoc_type = import_ref Main//a, loc17_9, loaded [concrete = constants.%assoc1.128]
// CHECK:STDOUT:   %Main.T.324 = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.737 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.a37 = import_ref Main//a, loc8_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5c = import_ref Main//a, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.e0c = import_ref Main//a, loc16_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.b1c = import_ref Main//a, loc17_9, unloaded
// CHECK:STDOUT:   %f_ref.patt: %pattern_type.afe = ref_binding_pattern f_ref [concrete]
// CHECK:STDOUT:   %f_ref.var_patt: %pattern_type.afe = var_pattern %f_ref.patt [concrete]
// CHECK:STDOUT:   %f_ref.var: ref %struct_type.f = var %f_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %e.patt: %pattern_type.7d1 = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.7d1 = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Empty.type = value_param call_param0
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty.type]
// CHECK:STDOUT:     %e: %Empty.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [concrete = constants.%UseBasic] {
// CHECK:STDOUT:     %e.patt: %pattern_type.185 = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.185 = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Basic.type = value_param call_param0
// CHECK:STDOUT:     %Basic.ref: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:     %e: %Basic.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [concrete = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %f.patt: %pattern_type.0a0 = value_binding_pattern f [concrete]
// CHECK:STDOUT:     %f.param_patt: %pattern_type.0a0 = value_param_pattern %f.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %f.param: %ForwardDeclared.type = value_param call_param0
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %f: %ForwardDeclared.type = value_binding f, %f.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %T.ref.loc10: %Basic.assoc_type = name_ref T, imports.%Main.import_ref.016 [concrete = constants.%assoc0.f14]
// CHECK:STDOUT:   %UseBasicT: %Basic.assoc_type = alias_binding UseBasicT, imports.%Main.import_ref.016 [concrete = constants.%assoc0.f14]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %F.ref.loc11: %Basic.assoc_type = name_ref F, imports.%Main.import_ref.f57 [concrete = constants.%assoc1.8cf]
// CHECK:STDOUT:   %UseBasicF: %Basic.assoc_type = alias_binding UseBasicF, imports.%Main.import_ref.f57 [concrete = constants.%assoc1.8cf]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %T.ref.loc13: %ForwardDeclared.assoc_type = name_ref T, imports.%Main.import_ref.d48 [concrete = constants.%assoc0.d49]
// CHECK:STDOUT:   %UseForwardDeclaredT: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredT, imports.%Main.import_ref.d48 [concrete = constants.%assoc0.d49]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %F.ref.loc14: %ForwardDeclared.assoc_type = name_ref F, imports.%Main.import_ref.176 [concrete = constants.%assoc1.128]
// CHECK:STDOUT:   %UseForwardDeclaredF: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredF, imports.%Main.import_ref.176 [concrete = constants.%assoc1.128]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.063 = ref_binding_pattern f [concrete]
// CHECK:STDOUT:     %f.var_patt: %pattern_type.063 = var_pattern %f.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %ptr.c31 = var %f.var_patt [concrete]
// CHECK:STDOUT:   %.loc16: type = splice_block %ptr [concrete = constants.%ptr.c31] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %ptr: type = ptr_type %ForwardDeclared.ref.loc16 [concrete = constants.%ptr.c31]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %ptr.c31 = ref_binding f, %f.var [concrete = %f.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.69f
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1e4
// CHECK:STDOUT:   .T = imports.%Main.import_ref.016
// CHECK:STDOUT:   .F = imports.%Main.import_ref.f57
// CHECK:STDOUT:   witness = (imports.%Main.T.faa, imports.%Main.F.dff)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.952
// CHECK:STDOUT:   .T = imports.%Main.import_ref.d48
// CHECK:STDOUT:   .F = imports.%Main.import_ref.176
// CHECK:STDOUT:   witness = (imports.%Main.T.324, imports.%Main.F.737)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e.param: %Empty.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e.param: %Basic.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
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
// CHECK:STDOUT:   %.loc16: ref %ForwardDeclared.type = struct_access %f_ref.ref, element0 [concrete = constants.%.816]
// CHECK:STDOUT:   %addr: %ptr.c31 = addr_of %.loc16 [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.5f3 = impl_witness_access constants.%Copy.impl_witness.e5d, element0 [concrete = constants.%ptr.as.Copy.impl.Op.a2e]
// CHECK:STDOUT:   %bound_method.loc16_27.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ForwardDeclared.type) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_27.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.c31 = call %bound_method.loc16_27.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%f.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
