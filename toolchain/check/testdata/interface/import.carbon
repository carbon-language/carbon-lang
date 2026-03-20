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
// CHECK:STDOUT:   %Self.61e: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.3e6: %Basic.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.590: %Basic.assoc_type = assoc_entity element0, @Basic.WithSelf.%T [concrete]
// CHECK:STDOUT:   %Basic.WithSelf.F.type: type = fn_type @Basic.WithSelf.F, @Basic.WithSelf(%Self.3e6) [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F: %Basic.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.6a5: %Basic.assoc_type = assoc_entity element1, @Basic.WithSelf.%Basic.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.8e7: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.e70: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.WithSelf.%T [concrete]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type: type = fn_type @ForwardDeclared.WithSelf.F, @ForwardDeclared.WithSelf(%Self.8e7) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F: %ForwardDeclared.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1.91f: %ForwardDeclared.assoc_type = assoc_entity element1, @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.527: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%struct_type.f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.8d4: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%struct_type.f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.100: %T.as.DefaultOrUnformed.impl.Op.type.8d4 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %struct_type.f, (%DefaultOrUnformed.impl_witness.527) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.100, @T.as.DefaultOrUnformed.impl.Op(%struct_type.f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %f_ref.patt: %pattern_type.afe = ref_binding_pattern f_ref [concrete]
// CHECK:STDOUT:     %f_ref.var_patt: %pattern_type.afe = var_pattern %f_ref.patt [concrete]
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
// CHECK:STDOUT:   %Self: %Basic.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3e6]
// CHECK:STDOUT:   %Basic.WithSelf.decl = interface_with_self_decl @Basic [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc8 [concrete] {
// CHECK:STDOUT:     %assoc0: %Basic.assoc_type = assoc_entity element0, @Basic.WithSelf.%T [concrete = constants.%assoc0.590]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.WithSelf.F.decl: @Basic.WithSelf.%Basic.WithSelf.F.type (%Basic.WithSelf.F.type) = fn_decl @Basic.WithSelf.F [symbolic = @Basic.WithSelf.%Basic.WithSelf.F (constants.%Basic.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc1: %Basic.assoc_type = assoc_entity element1, %Basic.WithSelf.F.decl [concrete = constants.%assoc1.6a5]
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
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8e7]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.decl = interface_with_self_decl @ForwardDeclared [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T.loc16 [concrete] {
// CHECK:STDOUT:     %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.WithSelf.%T [concrete = constants.%assoc0.e70]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.decl: @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.type (%ForwardDeclared.WithSelf.F.type) = fn_decl @ForwardDeclared.WithSelf.F [symbolic = @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F (constants.%ForwardDeclared.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc1: %ForwardDeclared.assoc_type = assoc_entity element1, %ForwardDeclared.WithSelf.F.decl [concrete = constants.%assoc1.91f]
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
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%struct_type.f, (constants.%DefaultOrUnformed.impl_witness.527) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc20_33.1: %DefaultOrUnformed.type = converted constants.%struct_type.f, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc20_33.1 [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   %.loc20_33.2: type = converted %.loc20_33.1, %as_type [concrete = constants.%struct_type.f]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.100, @T.as.DefaultOrUnformed.impl.Op(constants.%struct_type.f) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %struct_type.f = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%f_ref.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.61e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf(constants.%Self.3e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf.F(constants.%Self.3e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf(constants.%Self.8e7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.8e7
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type => constants.%ForwardDeclared.WithSelf.F.type
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F => constants.%ForwardDeclared.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf.F(constants.%Self.8e7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.0c3: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7d1: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [concrete]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.type: type = facet_type <@Basic> [concrete]
// CHECK:STDOUT:   %Self.685: %Basic.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F.type: type = fn_type @Basic.WithSelf.F, @Basic.WithSelf(%Self.685) [symbolic]
// CHECK:STDOUT:   %Basic.WithSelf.F: %Basic.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.185: type = pattern_type %Basic.type [concrete]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [concrete]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.332: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type: type = fn_type @ForwardDeclared.WithSelf.F, @ForwardDeclared.WithSelf(%Self.332) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F: %ForwardDeclared.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.0a0: type = pattern_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [concrete]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [concrete]
// CHECK:STDOUT:   %Basic.assoc_type: type = assoc_entity_type @Basic [concrete]
// CHECK:STDOUT:   %assoc0.a68: %Basic.assoc_type = assoc_entity element0, imports.%Main.import_ref.6c4 [concrete]
// CHECK:STDOUT:   %assoc1.448: %Basic.assoc_type = assoc_entity element1, imports.%Main.import_ref.f48 [concrete]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0.c28: %ForwardDeclared.assoc_type = assoc_entity element0, imports.%Main.import_ref.614 [concrete]
// CHECK:STDOUT:   %assoc1.e73: %ForwardDeclared.assoc_type = assoc_entity element1, imports.%Main.import_ref.e6e [concrete]
// CHECK:STDOUT:   %ptr.c31: type = ptr_type %ForwardDeclared.type [concrete]
// CHECK:STDOUT:   %pattern_type.063: type = pattern_type %ptr.c31 [concrete]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %ForwardDeclared.type} [concrete]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %struct_type.f [concrete]
// CHECK:STDOUT:   %.816: ref %ForwardDeclared.type = struct_access imports.%f_ref.var, element0 [concrete]
// CHECK:STDOUT:   %addr: %ptr.c31 = addr_of %.816 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e5d: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ba9: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ForwardDeclared.type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.a2e: %ptr.as.Copy.impl.Op.type.ba9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.c31, (%Copy.impl_witness.e5d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5bc: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.092: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5bc, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.736: %Basic.assoc_type = import_ref Main//a, loc8_8, loaded [concrete = constants.%assoc0.a68]
// CHECK:STDOUT:   %Main.import_ref.c83: %Basic.assoc_type = import_ref Main//a, loc9_9, loaded [concrete = constants.%assoc1.448]
// CHECK:STDOUT:   %Main.T.85a = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.95b = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.ba7f95.2: %Basic.type = import_ref Main//a, loc7_17, loaded [symbolic = constants.%Self.685]
// CHECK:STDOUT:   %Main.import_ref.1e4 = import_ref Main//a, loc7_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.fa6: %ForwardDeclared.assoc_type = import_ref Main//a, loc16_8, loaded [concrete = constants.%assoc0.c28]
// CHECK:STDOUT:   %Main.import_ref.9f2: %ForwardDeclared.assoc_type = import_ref Main//a, loc17_9, loaded [concrete = constants.%assoc1.e73]
// CHECK:STDOUT:   %Main.T.9f8 = import_ref Main//a, T, unloaded
// CHECK:STDOUT:   %Main.F.204 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.319eb9.2: %ForwardDeclared.type = import_ref Main//a, loc15_27, loaded [symbolic = constants.%Self.332]
// CHECK:STDOUT:   %Main.import_ref.952 = import_ref Main//a, loc15_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.6c4 = import_ref Main//a, loc8_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.f48 = import_ref Main//a, loc9_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.614 = import_ref Main//a, loc16_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.e6e = import_ref Main//a, loc17_9, unloaded
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
// CHECK:STDOUT:     %e.param_patt: %pattern_type.7d1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.7d1 = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Empty.type = value_param call_param0
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty.type]
// CHECK:STDOUT:     %e: %Empty.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [concrete = constants.%UseBasic] {
// CHECK:STDOUT:     %e.param_patt: %pattern_type.185 = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.185 = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %Basic.type = value_param call_param0
// CHECK:STDOUT:     %Basic.ref: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:     %e: %Basic.type = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [concrete = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %f.param_patt: %pattern_type.0a0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %f.patt: %pattern_type.0a0 = at_binding_pattern f, %f.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %f.param: %ForwardDeclared.type = value_param call_param0
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %f: %ForwardDeclared.type = value_binding f, %f.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %T.ref.loc10: %Basic.assoc_type = name_ref T, imports.%Main.import_ref.736 [concrete = constants.%assoc0.a68]
// CHECK:STDOUT:   %UseBasicT: %Basic.assoc_type = alias_binding UseBasicT, imports.%Main.import_ref.736 [concrete = constants.%assoc0.a68]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%Main.Basic [concrete = constants.%Basic.type]
// CHECK:STDOUT:   %F.ref.loc11: %Basic.assoc_type = name_ref F, imports.%Main.import_ref.c83 [concrete = constants.%assoc1.448]
// CHECK:STDOUT:   %UseBasicF: %Basic.assoc_type = alias_binding UseBasicF, imports.%Main.import_ref.c83 [concrete = constants.%assoc1.448]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %T.ref.loc13: %ForwardDeclared.assoc_type = name_ref T, imports.%Main.import_ref.fa6 [concrete = constants.%assoc0.c28]
// CHECK:STDOUT:   %UseForwardDeclaredT: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredT, imports.%Main.import_ref.fa6 [concrete = constants.%assoc0.c28]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %F.ref.loc14: %ForwardDeclared.assoc_type = name_ref F, imports.%Main.import_ref.9f2 [concrete = constants.%assoc1.e73]
// CHECK:STDOUT:   %UseForwardDeclaredF: %ForwardDeclared.assoc_type = alias_binding UseForwardDeclaredF, imports.%Main.import_ref.9f2 [concrete = constants.%assoc1.e73]
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
// CHECK:STDOUT:   .T = imports.%Main.import_ref.736
// CHECK:STDOUT:   .F = imports.%Main.import_ref.c83
// CHECK:STDOUT:   witness = (imports.%Main.T.85a, imports.%Main.F.95b)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.952
// CHECK:STDOUT:   .T = imports.%Main.import_ref.fa6
// CHECK:STDOUT:   .F = imports.%Main.import_ref.9f2
// CHECK:STDOUT:   witness = (imports.%Main.T.9f8, imports.%Main.F.204)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e.param: %Empty.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Basic.WithSelf.F(imports.%Main.import_ref.ba7f95.2: %Basic.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e.param: %Basic.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared.WithSelf.F(imports.%Main.import_ref.319eb9.2: %ForwardDeclared.type) [from "a.carbon"] {
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
// CHECK:STDOUT:   %.loc16: ref %ForwardDeclared.type = struct_access %f_ref.ref, element0 [concrete = constants.%.816]
// CHECK:STDOUT:   %addr: %ptr.c31 = addr_of %.loc16 [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.092 = impl_witness_access constants.%Copy.impl_witness.e5d, element0 [concrete = constants.%ptr.as.Copy.impl.Op.a2e]
// CHECK:STDOUT:   %bound_method.loc16_27.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ForwardDeclared.type) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_27.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.c31 = call %bound_method.loc16_27.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%f.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.0c3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf(constants.%Self.685) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.685
// CHECK:STDOUT:   %Basic.WithSelf.F.type => constants.%Basic.WithSelf.F.type
// CHECK:STDOUT:   %Basic.WithSelf.F => constants.%Basic.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic.WithSelf.F(constants.%Self.685) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf(constants.%Self.332) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.332
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type => constants.%ForwardDeclared.WithSelf.F.type
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F => constants.%ForwardDeclared.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf.F(constants.%Self.332) {}
// CHECK:STDOUT:
