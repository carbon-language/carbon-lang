// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/import.carbon

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
// CHECK:STDOUT:   %Self.193: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.1c7: %Basic.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type %Basic.type [concrete]
// CHECK:STDOUT:   %assoc0.017: %Basic.assoc_type = assoc_entity element0, @Basic.%T [concrete]
// CHECK:STDOUT:   %F.type.320: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.952: %F.type.320 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.9c9: %Basic.assoc_type = assoc_entity element1, @Basic.%F.decl [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.efa: %ForwardDeclared.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %assoc0.ec6: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.%T [concrete]
// CHECK:STDOUT:   %F.type.505: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.eb5: %F.type.505 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.02a: %ForwardDeclared.assoc_type = assoc_entity element1, @ForwardDeclared.%F.decl [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Basic = %Basic.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl
// CHECK:STDOUT:     .f_ref = %f_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %Basic.decl: type = interface_decl @Basic [concrete = constants.%Basic.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl: type = interface_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f_ref.patt: %struct_type.f = binding_pattern f_ref
// CHECK:STDOUT:     %.loc20_1: %struct_type.f = var_pattern %f_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f_ref.var: ref %struct_type.f = var f_ref
// CHECK:STDOUT:   %.loc20_32: type = splice_block %struct_type.f [concrete = constants.%struct_type.f] {
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f_ref: ref %struct_type.f = bind_name f_ref, %f_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.193]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %Self: %Basic.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1c7]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc8 [concrete] {
// CHECK:STDOUT:     %assoc0: %Basic.assoc_type = assoc_entity element0, @Basic.%T [concrete = constants.%assoc0.017]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.320 = fn_decl @F.1 [concrete = constants.%F.952] {} {}
// CHECK:STDOUT:   %assoc1: %Basic.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.9c9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc8.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.efa]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc16 [concrete] {
// CHECK:STDOUT:     %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.%T [concrete = constants.%assoc0.ec6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.505 = fn_decl @F.2 [concrete = constants.%F.eb5] {} {}
// CHECK:STDOUT:   %assoc1: %ForwardDeclared.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1.02a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.loc16.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %F.decl)
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
// CHECK:STDOUT: generic fn @F.1(@Basic.%Self: %Basic.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.loc8(constants.%Self.1c7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1c7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.loc16(constants.%Self.efa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.efa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [concrete]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [concrete]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [concrete]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type %Basic.type [concrete]
// CHECK:STDOUT:   %assoc0.b26: %Basic.assoc_type = assoc_entity element0, imports.%Main.import_ref.a4a [concrete]
// CHECK:STDOUT:   %assoc1.0bf: %Basic.assoc_type = assoc_entity element1, imports.%Main.import_ref.0be [concrete]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %assoc0.c94: %ForwardDeclared.assoc_type = assoc_entity element0, imports.%Main.import_ref.69a [concrete]
// CHECK:STDOUT:   %assoc1.660: %ForwardDeclared.assoc_type = assoc_entity element1, imports.%Main.import_ref.1cc [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Empty: type = import_ref Main//a, Empty, loaded [concrete = constants.%Empty.type]
// CHECK:STDOUT:   %Main.Basic: type = import_ref Main//a, Basic, loaded [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %Main.ForwardDeclared: type = import_ref Main//a, ForwardDeclared, loaded [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %Main.f_ref: ref %struct_type.f = import_ref Main//a, f_ref, loaded
// CHECK:STDOUT:   %Main.import_ref.cc0 = import_ref Main//a, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.37f = import_ref Main//a, inst20 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.152: %Basic.assoc_type = import_ref Main//a, loc8_8, loaded [concrete = constants.%assoc0.b26]
// CHECK:STDOUT:   %Main.import_ref.30b: %Basic.assoc_type = import_ref Main//a, loc9_9, loaded [concrete = constants.%assoc1.0bf]
// CHECK:STDOUT:   %Main.T.44f = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.eea = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.52b = import_ref Main//a, inst34 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.de0: %ForwardDeclared.assoc_type = import_ref Main//a, loc16_8, loaded [concrete = constants.%assoc0.c94]
// CHECK:STDOUT:   %Main.import_ref.c9c: %ForwardDeclared.assoc_type = import_ref Main//a, loc17_9, loaded [concrete = constants.%assoc1.660]
// CHECK:STDOUT:   %Main.T.6ee = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.5d0 = import_ref Main//a, F, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = imports.%Main.Empty
// CHECK:STDOUT:     .Basic = imports.%Main.Basic
// CHECK:STDOUT:     .ForwardDeclared = imports.%Main.ForwardDeclared
// CHECK:STDOUT:     .f_ref = imports.%Main.f_ref
// CHECK:STDOUT:     .UseEmpty = %UseEmpty.decl
// CHECK:STDOUT:     .UseBasic = %UseBasic.decl
// CHECK:STDOUT:     .UseForwardDeclared = %UseForwardDeclared.decl
// CHECK:STDOUT:     .UseBasicT = %UseBasicT
// CHECK:STDOUT:     .UseBasicF = %UseBasicF
// CHECK:STDOUT:     .UseForwardDeclaredT = %UseForwardDeclaredT
// CHECK:STDOUT:     .UseForwardDeclaredF = %UseForwardDeclaredF
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %UseEmpty.decl: %UseEmpty.type = fn_decl @UseEmpty [concrete = constants.%UseEmpty] {
// CHECK:STDOUT:     %e.patt: %Empty.type = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %Empty.type = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Empty.type = value_param call_param0
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty.type]
// CHECK:STDOUT:     %e: %Empty.type = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [concrete = constants.%UseBasic] {
// CHECK:STDOUT:     %e.patt: %Basic.type = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %Basic.type = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Basic.type = value_param call_param0
// CHECK:STDOUT:     %Basic.ref: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:     %e: %Basic.type = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [concrete = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %f.patt: %ForwardDeclared.type = binding_pattern f
// CHECK:STDOUT:     %f.param_patt: %ForwardDeclared.type = value_param_pattern %f.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %f.param: %ForwardDeclared.type = value_param call_param0
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %f: %ForwardDeclared.type = bind_name f, %f.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %T.ref.loc10: %Basic.assoc_type = name_ref T, imports.%Main.import_ref.152 [concrete = constants.%assoc0.b26]
// CHECK:STDOUT:   %UseBasicT: %Basic.assoc_type = bind_alias UseBasicT, imports.%Main.import_ref.152 [concrete = constants.%assoc0.b26]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %F.ref.loc11: %Basic.assoc_type = name_ref F, imports.%Main.import_ref.30b [concrete = constants.%assoc1.0bf]
// CHECK:STDOUT:   %UseBasicF: %Basic.assoc_type = bind_alias UseBasicF, imports.%Main.import_ref.30b [concrete = constants.%assoc1.0bf]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %T.ref.loc13: %ForwardDeclared.assoc_type = name_ref T, imports.%Main.import_ref.de0 [concrete = constants.%assoc0.c94]
// CHECK:STDOUT:   %UseForwardDeclaredT: %ForwardDeclared.assoc_type = bind_alias UseForwardDeclaredT, imports.%Main.import_ref.de0 [concrete = constants.%assoc0.c94]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %F.ref.loc14: %ForwardDeclared.assoc_type = name_ref F, imports.%Main.import_ref.c9c [concrete = constants.%assoc1.660]
// CHECK:STDOUT:   %UseForwardDeclaredF: %ForwardDeclared.assoc_type = bind_alias UseForwardDeclaredF, imports.%Main.import_ref.c9c [concrete = constants.%assoc1.660]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %ptr = binding_pattern f
// CHECK:STDOUT:     %.loc16_1: %ptr = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %ptr = var f
// CHECK:STDOUT:   %.loc16_23: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %ptr: type = ptr_type %ForwardDeclared.ref.loc16 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %ptr = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cc0
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.37f
// CHECK:STDOUT:   .T = imports.%Main.import_ref.152
// CHECK:STDOUT:   .F = imports.%Main.import_ref.30b
// CHECK:STDOUT:   witness = (imports.%Main.T.44f, imports.%Main.F.eea)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.52b
// CHECK:STDOUT:   .T = imports.%Main.import_ref.de0
// CHECK:STDOUT:   .F = imports.%Main.import_ref.c9c
// CHECK:STDOUT:   witness = (imports.%Main.T.6ee, imports.%Main.F.5d0)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e.param_patt: %Empty.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e.param_patt: %Basic.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f.param_patt: %ForwardDeclared.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref %struct_type.f = name_ref f_ref, imports.%Main.f_ref
// CHECK:STDOUT:   %.loc16: ref %ForwardDeclared.type = struct_access %f_ref.ref, element0
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc16
// CHECK:STDOUT:   assign file.%f.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
