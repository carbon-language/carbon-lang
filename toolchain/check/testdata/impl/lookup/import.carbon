// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/import.carbon

// --- package_a.carbon

package PackageA;

interface HasF {
  fn F[self: Self]();
}

class C {}

// Same library as the class and interface.
impl C as HasF {
  fn F[self: Self]() {}
}

// --- package_b.carbon

package PackageB;

import PackageA;

interface HasG {
  fn G[self: Self]();
}

class D {}

// Same library as the interface.
impl PackageA.C as HasG {
  fn G[self: Self]() {}
}

// Same library as the class.
impl D as PackageA.HasF {
  fn F[self: Self]() {}
}

// Same library as the class and interface.
impl D as HasG {
  fn G[self: Self]() {}
}

// --- use_cf.carbon

library "[[@TEST_NAME]]";

import PackageA;

fn TestCF(c: PackageA.C) {
  c.(PackageA.HasF.F)();
}

// --- use_df.carbon

library "[[@TEST_NAME]]";

import PackageA;
import PackageB;

fn TestDF(d: PackageB.D) {
  d.(PackageA.HasF.F)();
}


// --- use_cg.carbon

library "[[@TEST_NAME]]";

import PackageA;
import PackageB;

fn TestCG(c: PackageA.C) {
  c.(PackageB.HasG.G)();
}

// --- use_dg.carbon

library "[[@TEST_NAME]]";

import PackageB;

fn TestDG(d: PackageB.D) {
  d.(PackageB.HasG.G)();
}

// --- associated_interface.carbon

package PackageAssociatedInterface;

interface Z {
  fn H[self: Self]();
}

impl () as Z {
  fn H[self: Self]() {}
}

// --- import_associated_interface.carbon

package PackageImportAssociatedInterface;

import PackageAssociatedInterface;

fn J() {
  ().(PackageAssociatedInterface.Z.H)();
}

// --- has_param.carbon

package PackageHasParam;

class AnyParam[T:! type](X:! T) {}

interface Y {
  fn K[self: Self]() {}
}

// --- has_generic_interface.carbon

package PackageGenericInterface;

import PackageHasParam;

interface GenericInterface(U:! type) {}

impl PackageHasParam.AnyParam(GenericInterface) as PackageHasParam.Y {
  fn K[self: Self]() {}
}

fn L() {
  var obj: PackageHasParam.AnyParam(GenericInterface) = {};
  obj.(PackageHasParam.Y.K)();
}

// --- use_generic_interface_as_param.carbon

library "[[@TEST_NAME]]";

import PackageHasParam;
import PackageGenericInterface;

fn M() {
  var obj: PackageHasParam.AnyParam(
    PackageGenericInterface.GenericInterface) = {};
  obj.(PackageHasParam.Y.K)();
}

// --- has_generic_class.carbon

package PackageGenericClass;

import PackageHasParam;

class GenericClass(U:! type) {}

impl PackageHasParam.AnyParam(GenericClass) as PackageHasParam.Y {
  fn K[self: Self]() {}
}

fn L() {
  var obj: PackageHasParam.AnyParam(GenericClass) = {};
  obj.(PackageHasParam.Y.K)();
}

// --- use_generic_class_as_param.carbon

library "[[@TEST_NAME]]";

import PackageHasParam;
import PackageGenericClass;

fn M() {
  var obj: PackageHasParam.AnyParam(PackageGenericClass.GenericClass) = {};
  obj.(PackageHasParam.Y.K)();
}

// --- has_extra_interfaces.carbon

package HasExtraInterfaces;

interface Extra1 {}
interface Extra2 {}
interface Extra3 {}
interface Extra4 {}
interface Extra5 {}
interface Extra6 {}
interface Extra7 {}
interface Extra8 {}

class C(T:! type) {}
interface I { fn F[self: Self](); }

impl C((Extra1, Extra2, Extra3, Extra4, Extra5, Extra6, Extra7, Extra8)) as I {
  fn F[self: Self]() {}
}

// --- fail_use_has_extra_interfaces.carbon

package UseHasExtraInterfaces;
import HasExtraInterfaces;

fn Test(c: HasExtraInterfaces.C(type)) {
  // This triggers the import of a bunch more interfaces, which reallocates the
  // interface ValueStore. Ensure that doesn't result in a use-after-free crash.
  // CHECK:STDERR: fail_use_has_extra_interfaces.carbon:[[@LINE+4]]:3: error: cannot access member of interface `HasExtraInterfaces.I` in type `HasExtraInterfaces.C(type)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c.(HasExtraInterfaces.I.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.(HasExtraInterfaces.I.F)();
}

// CHECK:STDOUT: --- package_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.85b: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%HasF.impl_witness) [concrete]
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
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@C.as.HasF.impl.%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.85b) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.85b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %C.ref as %HasF.ref {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = file.%HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.85b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.85b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.4d5: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.5b9: type = facet_access_type %Self.4d5 [symbolic]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %Self.as_type.5b9 [symbolic]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, @HasG.%HasG.G.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.c63: <witness> = impl_witness file.%HasG.impl_witness_table.loc13 [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G.type: type = fn_type @C.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G: %C.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.a10: %HasG.type = facet_value %C, (%HasG.impl_witness.c63) [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.d86: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.a8f: type = facet_access_type %Self.d86 [symbolic]
// CHECK:STDOUT:   %pattern_type.fcd: type = pattern_type %Self.as_type.a8f [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness file.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F.type: type = fn_type @D.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F: %D.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.c28: <witness> = impl_witness file.%HasG.impl_witness_table.loc23 [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G.type: type = fn_type @D.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G: %D.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.6ec: %HasG.type = facet_value %D, (%HasG.impl_witness.c28) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .C = %PackageA.C
// CHECK:STDOUT:     .HasF = %PackageA.HasF
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.C: type = import_ref PackageA//default, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst43 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.848 = import_ref PackageA//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.c63 = import_ref PackageA//default, loc5_21, unloaded
// CHECK:STDOUT:   %PackageA.F: %HasF.F.type = import_ref PackageA//default, F, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.0b6: %HasF.type = import_ref PackageA//default, inst19 [no loc], loaded [symbolic = constants.%Self.d86]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [concrete = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @C.as.HasG.impl [concrete] {} {
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasG.impl_witness_table.loc13 = impl_witness_table (@C.as.HasG.impl.%C.as.HasG.impl.G.decl), @C.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.loc13: <witness> = impl_witness %HasG.impl_witness_table.loc13 [concrete = constants.%HasG.impl_witness.c63]
// CHECK:STDOUT:   impl_decl @D.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@D.as.HasF.impl.%D.as.HasF.impl.F.decl), @D.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   impl_decl @D.as.HasG.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasG.impl_witness_table.loc23 = impl_witness_table (@D.as.HasG.impl.%D.as.HasG.impl.G.decl), @D.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.loc23: <witness> = impl_witness %HasG.impl_witness_table.loc23 [concrete = constants.%HasG.impl_witness.c28]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4d5]
// CHECK:STDOUT:   %HasG.G.decl: %HasG.G.type = fn_decl @HasG.G [concrete = constants.%HasG.G] {
// CHECK:STDOUT:     %self.patt: @HasG.G.%pattern_type (%pattern_type.511) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasG.G.%pattern_type (%pattern_type.511) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasG.G.%Self.as_type.loc7_14.1 (%Self.as_type.5b9) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.5b9)] {
// CHECK:STDOUT:       %Self.ref: %HasG.type = name_ref Self, @HasG.%Self [symbolic = %Self (constants.%Self.4d5)]
// CHECK:STDOUT:       %Self.as_type.loc7_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.5b9)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.5b9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasG.G.%Self.as_type.loc7_14.1 (%Self.as_type.5b9) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %HasG.G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%HasG.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.848
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.c63
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: %C.ref as %HasG.ref {
// CHECK:STDOUT:   %C.as.HasG.impl.G.decl: %C.as.HasG.impl.G.type = fn_decl @C.as.HasG.impl.G [concrete = constants.%C.as.HasG.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8e5 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8e5 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.HasG.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.HasG.impl.G.decl
// CHECK:STDOUT:   witness = file.%HasG.impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasF.impl: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %D.as.HasF.impl.F.decl: %D.as.HasF.impl.F.type = fn_decl @D.as.HasF.impl.F [concrete = constants.%D.as.HasF.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @D.as.HasF.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = file.%HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %D.as.HasG.impl.G.decl: %D.as.HasG.impl.G.type = fn_decl @D.as.HasG.impl.G [concrete = constants.%D.as.HasG.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @D.as.HasG.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %D.as.HasG.impl.G.decl
// CHECK:STDOUT:   witness = file.%HasG.impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.4d5)]
// CHECK:STDOUT:   %Self.as_type.loc7_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.5b9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc7_14.1 [symbolic = %pattern_type (constants.%pattern_type.511)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasG.G.%Self.as_type.loc7_14.1 (%Self.as_type.5b9));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.0b6: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.d86)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.a8f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.fcd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasF.impl.F(%self.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasG.impl.G(%self.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self.4d5) {
// CHECK:STDOUT:   %Self => constants.%Self.4d5
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%Self.as_type.5b9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.511
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%HasG.facet.a10) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.a10
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self.d86) {
// CHECK:STDOUT:   %Self => constants.%Self.d86
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.a8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%HasG.facet.6ec) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.6ec
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.ab2 [concrete]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.fcd: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness imports.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %.3f2: type = fn_type_with_self_type %HasF.F.type, %HasF.facet [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .C = %PackageA.C
// CHECK:STDOUT:     .HasF = %PackageA.HasF
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.C: type = import_ref PackageA//default, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst43 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.848 = import_ref PackageA//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.b36: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.ab2: %HasF.F.type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.0b6: %HasF.type = import_ref PackageA//default, inst19 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageA.import_ref.a12: <witness> = import_ref PackageA//default, loc11_16, loaded [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   %PackageA.import_ref.29a: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.e8c: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.68f: %C.as.HasF.impl.F.type = import_ref PackageA//default, loc12_22, loaded [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%PackageA.import_ref.68f), @C.as.HasF.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .TestCF = %TestCF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %TestCF.decl: %TestCF.type = fn_decl @TestCF [concrete = constants.%TestCF] {
// CHECK:STDOUT:     %c.patt: %pattern_type.8e5 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.8e5 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref.loc6: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.848
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.b36
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: imports.%PackageA.import_ref.29a as imports.%PackageA.import_ref.e8c [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.a12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc7: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.b36 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.3f2 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.0b6: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.fcd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F [from "package_a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_df.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f09: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.ab2 [concrete]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.fcd: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness imports.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %.321: type = fn_type_with_self_type %HasF.F.type, %HasF.facet [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F.type: type = fn_type @D.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F: %D.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .HasF = %PackageA.HasF
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [concrete] {
// CHECK:STDOUT:     .D = %PackageB.D
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB.D: type = import_ref PackageB//default, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst45 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.848 = import_ref PackageA//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.b36: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.ab2: %HasF.F.type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.0b6: %HasF.type = import_ref PackageA//default, inst19 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageA.import_ref.54d = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst43 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.29a: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.e8c: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.5e6: <witness> = import_ref PackageB//default, loc18_25, loaded [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.aa9: type = import_ref PackageB//default, loc18_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.831: type = import_ref PackageB//default, loc18_19, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.9f3: %D.as.HasF.impl.F.type = import_ref PackageB//default, loc19_22, loaded [concrete = constants.%D.as.HasF.impl.F]
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%PackageB.import_ref.9f3), @D.as.HasF.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDF = %TestDF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDF.decl: %TestDF.type = fn_decl @TestDF [concrete = constants.%TestDF] {
// CHECK:STDOUT:     %d.patt: %pattern_type.f09 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.f09 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%PackageB.D [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.848
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.b36
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: imports.%PackageA.import_ref.29a as imports.%PackageA.import_ref.e8c [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.54d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasF.impl: imports.%PackageB.import_ref.aa9 as imports.%PackageB.import_ref.831 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.5e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.cab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDF(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.b36 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.321 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%D.as.HasF.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %D.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.0b6: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.fcd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasF.impl.F [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.70a [concrete]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.bf4: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness imports.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %.f91: type = fn_type_with_self_type %HasG.G.type, %HasG.facet [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G.type: type = fn_type @C.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G: %C.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .C = %PackageA.C
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [concrete] {
// CHECK:STDOUT:     .HasG = %PackageB.HasG
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.C: type = import_ref PackageA//default, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst43 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.c0d = import_ref PackageB//default, inst21 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.6c2: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.70a: %HasG.G.type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%HasG.G]
// CHECK:STDOUT:   %PackageB.import_ref.4b1: %HasG.type = import_ref PackageB//default, inst21 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageB.import_ref.f06: <witness> = import_ref PackageB//default, loc13_25, loaded [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.dfb: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.25f = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst45 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.aa9: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.872: %C.as.HasG.impl.G.type = import_ref PackageB//default, loc14_22, loaded [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%PackageB.import_ref.872), @C.as.HasG.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestCG = %TestCG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestCG.decl: %TestCG.type = fn_decl @TestCG [concrete = constants.%TestCG] {
// CHECK:STDOUT:     %c.patt: %pattern_type.8e5 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.8e5 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.c0d
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.6c2
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: imports.%PackageB.import_ref.dfb as imports.%PackageB.import_ref.cee586.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.f06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: imports.%PackageB.import_ref.aa9 as imports.%PackageB.import_ref.cee586.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.25f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.cab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCG(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.6c2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.f91 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.HasG.impl.G.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(imports.%PackageB.import_ref.4b1: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.bf4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bf4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_dg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f09: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.70a [concrete]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.bf4: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness imports.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %D, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %.3e2: type = fn_type_with_self_type %HasG.G.type, %HasG.facet [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G.type: type = fn_type @D.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G: %D.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [concrete] {
// CHECK:STDOUT:     .D = %PackageB.D
// CHECK:STDOUT:     .HasG = %PackageB.HasG
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB.D: type = import_ref PackageB//default, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst45 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.c0d = import_ref PackageB//default, inst21 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.6c2: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.70a: %HasG.G.type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%HasG.G]
// CHECK:STDOUT:   %PackageB.import_ref.4b1: %HasG.type = import_ref PackageB//default, inst21 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageB.import_ref.f94 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8db: <witness> = import_ref PackageB//default, inst53 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.6a9 = import_ref PackageB//default, inst54 [indirect], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.dfb: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.179: <witness> = import_ref PackageB//default, loc23_16, loaded [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.aa9: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.ad0: %D.as.HasG.impl.G.type = import_ref PackageB//default, loc24_22, loaded [concrete = constants.%D.as.HasG.impl.G]
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%PackageB.import_ref.ad0), @D.as.HasG.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDG = %TestDG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDG.decl: %TestDG.type = fn_decl @TestDG [concrete = constants.%TestDG] {
// CHECK:STDOUT:     %d.patt: %pattern_type.f09 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.f09 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref.loc6: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%PackageB.D [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.c0d
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.6c2
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: imports.%PackageB.import_ref.dfb as imports.%PackageB.import_ref.cee586.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.f94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: imports.%PackageB.import_ref.aa9 as imports.%PackageB.import_ref.cee586.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.cab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDG(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc7: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.6c2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.3e2 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%D.as.HasG.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %D.as.HasG.impl.G.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(imports.%PackageB.import_ref.4b1: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.bf4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasG.impl.G [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bf4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.c8d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Z.H.type: type = fn_type @Z.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.H: %Z.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%Z.H.decl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.type: type = fn_type @empty_tuple.type.as.Z.impl.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H: %empty_tuple.type.as.Z.impl.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%Z.impl_witness) [concrete]
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
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (@empty_tuple.type.as.Z.impl.%empty_tuple.type.as.Z.impl.H.decl), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z.H.decl: %Z.H.type = fn_decl @Z.H [concrete = constants.%Z.H] {
// CHECK:STDOUT:     %self.patt: @Z.H.%pattern_type (%pattern_type.c8d) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Z.H.%pattern_type (%pattern_type.c8d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Z.H.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Z.H.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.H.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %assoc0
// CHECK:STDOUT:   witness = (%Z.H.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.decl: %empty_tuple.type.as.Z.impl.H.type = fn_decl @empty_tuple.type.as.Z.impl.H [concrete = constants.%empty_tuple.type.as.Z.impl.H] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.Z.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %empty_tuple.type.as.Z.impl.H.decl
// CHECK:STDOUT:   witness = file.%Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.H(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type (constants.%pattern_type.c8d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Z.H.%Self.as_type.loc5_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.Z.impl.H(%self.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c8d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, imports.%PackageAssociatedInterface.import_ref.250 [concrete]
// CHECK:STDOUT:   %Z.H.type: type = fn_type @Z.H [concrete]
// CHECK:STDOUT:   %Z.H: %Z.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6b0: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness imports.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %.a1b: type = fn_type_with_self_type %Z.H.type, %Z.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.type: type = fn_type @empty_tuple.type.as.Z.impl.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H: %empty_tuple.type.as.Z.impl.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageAssociatedInterface: <namespace> = namespace file.%PackageAssociatedInterface.import, [concrete] {
// CHECK:STDOUT:     .Z = %PackageAssociatedInterface.Z
// CHECK:STDOUT:     import PackageAssociatedInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageAssociatedInterface.Z: type = import_ref PackageAssociatedInterface//default, Z, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.362 = import_ref PackageAssociatedInterface//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.609: %Z.assoc_type = import_ref PackageAssociatedInterface//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageAssociatedInterface.H = import_ref PackageAssociatedInterface//default, H, unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.250: %Z.H.type = import_ref PackageAssociatedInterface//default, loc5_21, loaded [concrete = constants.%Z.H]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.863: %Z.type = import_ref PackageAssociatedInterface//default, inst19 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.915: <witness> = import_ref PackageAssociatedInterface//default, loc8_14, loaded [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.e5c: type = import_ref PackageAssociatedInterface//default, loc8_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.df1: type = import_ref PackageAssociatedInterface//default, loc8_12, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.108: %empty_tuple.type.as.Z.impl.H.type = import_ref PackageAssociatedInterface//default, loc9_22, loaded [concrete = constants.%empty_tuple.type.as.Z.impl.H]
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%PackageAssociatedInterface.import_ref.108), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageAssociatedInterface = imports.%PackageAssociatedInterface
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageAssociatedInterface.import = import PackageAssociatedInterface
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageAssociatedInterface.import_ref.362
// CHECK:STDOUT:   .H = imports.%PackageAssociatedInterface.import_ref.609
// CHECK:STDOUT:   witness = (imports.%PackageAssociatedInterface.H)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: imports.%PackageAssociatedInterface.import_ref.e5c as imports.%PackageAssociatedInterface.import_ref.df1 [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageAssociatedInterface.import_ref.915
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_4.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %PackageAssociatedInterface.ref: <namespace> = name_ref PackageAssociatedInterface, imports.%PackageAssociatedInterface [concrete = imports.%PackageAssociatedInterface]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, imports.%PackageAssociatedInterface.Z [concrete = constants.%Z.type]
// CHECK:STDOUT:   %H.ref: %Z.assoc_type = name_ref H, imports.%PackageAssociatedInterface.import_ref.609 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.a1b = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.Z.impl.H]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc7_4.1, %impl.elem0
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_4.2: %empty_tuple.type = converted %.loc7_4.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.call: init %empty_tuple.type = call %bound_method(%.loc7_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.H(imports.%PackageAssociatedInterface.import_ref.863: %Z.type) [from "associated_interface.carbon"] {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.6b0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.Z.impl.H [from "associated_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.5da: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y.K.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
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
// CHECK:STDOUT:     .AnyParam = %AnyParam.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AnyParam.decl: %AnyParam.type = class_decl @AnyParam [concrete = constants.%AnyParam.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %X.patt: @AnyParam.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern X, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4: type = splice_block %T.ref [symbolic = %T.loc4_16.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_16.2 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc4_26.2: @AnyParam.%T.loc4_16.1 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc4_26.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Y.K.decl: %Y.K.type = fn_decl @Y.K [concrete = constants.%Y.K] {
// CHECK:STDOUT:     %self.patt: @Y.K.%pattern_type (%pattern_type.5da) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Y.K.%pattern_type (%pattern_type.5da) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Y.K.%Self.as_type.loc7_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Y.type = name_ref Self, @Y.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc7_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Y.K.%Self.as_type.loc7_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, %Y.K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%Y.K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(%T.loc4_16.2: type, %X.loc4_26.2: @AnyParam.%T.loc4_16.1 (%T)) {
// CHECK:STDOUT:   %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   %X.loc4_26.1: @AnyParam.%T.loc4_16.1 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc4_26.1 (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_16.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%AnyParam
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(@Y.%Self: %Y.type) {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc7_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc7_14.1 [symbolic = %pattern_type (constants.%pattern_type.5da)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc7_14.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Y.K.%Self.as_type.loc7_14.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:   %X.loc4_26.1 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericInterface.type.c92: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.c92 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericInterface.type.0d7: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.185: %GenericInterface.type.0d7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.438: type = pattern_type %GenericInterface.type.c92 [concrete]
// CHECK:STDOUT:   %AnyParam.241: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.c92, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.2f4: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.84b: type = facet_access_type %Self.2f4 [symbolic]
// CHECK:STDOUT:   %pattern_type.e57: type = pattern_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %require_complete.040: <witness> = require_complete_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.004: type = pattern_type %AnyParam.241 [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.241, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.241 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.494: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %.41c: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AnyParam.241, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.925: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.290: %AggregateT.as_type.as.Destroy.impl.Op.type.925 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %AnyParam.241 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.290, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst38 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.0f3 = import_ref PackageHasParam//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.f69: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %PackageHasParam.K: %Y.K.type = import_ref PackageHasParam//default, K, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.28c: %Y.type = import_ref PackageHasParam//default, inst44 [no loc], loaded [symbolic = constants.%Self.2f4]
// CHECK:STDOUT:   %PackageHasParam.import_ref.ce2: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.c92 = interface_decl @GenericInterface [concrete = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc6_28.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @AnyParam.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.c92 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.241]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_52: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (@AnyParam.as.Y.impl.%AnyParam.as.Y.impl.K.decl), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%U.loc6_28.2: type) {
// CHECK:STDOUT:   %U.loc6_28.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U.loc6_28.1)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.0d7)]
// CHECK:STDOUT:   %Self.2: @GenericInterface.%GenericInterface.type (%GenericInterface.type.0d7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.185)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.0d7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.185)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.0f3
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.f69
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.decl: %AnyParam.as.Y.impl.K.type = fn_decl @AnyParam.as.Y.impl.K [concrete = constants.%AnyParam.as.Y.impl.K] {
// CHECK:STDOUT:     %self.patt: %pattern_type.004 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.004 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.241 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @AnyParam.as.Y.impl.%AnyParam [concrete = constants.%AnyParam.241]
// CHECK:STDOUT:     %self: %AnyParam.241 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %AnyParam.as.Y.impl.K.decl
// CHECK:STDOUT:   witness = file.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.601
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.28c: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2f4)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.84b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.e57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic = %require_complete (constants.%require_complete.040)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K(%self.param: %AnyParam.241) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.004 = binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.004 = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.241 = var %obj.var_patt
// CHECK:STDOUT:   %.loc13_58.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_58.2: init %AnyParam.241 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3.1: init %AnyParam.241 = converted %.loc13_58.1, %.loc13_58.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3.1
// CHECK:STDOUT:   %.loc13_53: type = splice_block %AnyParam [concrete = constants.%AnyParam.241] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.c92 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.241]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.241 = bind_name obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.241 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.f69 [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %impl.elem0: %.41c = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.241 = bind_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method.loc14(%.loc14)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AnyParam.241, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc13_3.2: %type_where = converted constants.%AnyParam.241, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %obj.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.290
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.290, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %obj.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %obj.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.c92
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.438
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.2f4) {
// CHECK:STDOUT:   %Self => constants.%Self.2f4
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.84b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.as_type => constants.%AnyParam.241
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.004
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_interface_as_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.0da: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.55d: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.719: %GenericInterface.type.55d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.a64: type = pattern_type %GenericInterface.type.0da [concrete]
// CHECK:STDOUT:   %AnyParam.861: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.0da, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %pattern_type.3ad: type = pattern_type %AnyParam.861 [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.861 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.2f4: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.494: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.84b: type = facet_access_type %Self.2f4 [symbolic]
// CHECK:STDOUT:   %pattern_type.e57: type = pattern_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %require_complete.040: <witness> = require_complete_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness imports.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.861, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %.ed5: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AnyParam.861, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0b7: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.596: %AggregateT.as_type.as.Destroy.impl.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1ef: type = ptr_type %AnyParam.861 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.596, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageGenericInterface: <namespace> = namespace file.%PackageGenericInterface.import, [concrete] {
// CHECK:STDOUT:     .GenericInterface = %PackageGenericInterface.GenericInterface
// CHECK:STDOUT:     import PackageGenericInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst38 [no loc], unloaded
// CHECK:STDOUT:   %PackageGenericInterface.GenericInterface: %GenericInterface.type.0da = import_ref PackageGenericInterface//default, GenericInterface, loaded [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.5ab: type = import_ref PackageGenericInterface//default, loc6_28, loaded [symbolic = @GenericInterface.%U (constants.%U)]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.b37 = import_ref PackageGenericInterface//default, inst32 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.0f3 = import_ref PackageHasParam//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.f69: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.ce2: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.28c: %Y.type = import_ref PackageHasParam//default, inst44 [no loc], loaded [symbolic = constants.%Self.2f4]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.ba6: <witness> = import_ref PackageGenericInterface//default, loc8_70, loaded [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.321: type = import_ref PackageGenericInterface//default, loc8_47, loaded [concrete = constants.%AnyParam.861]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.ca6: type = import_ref PackageGenericInterface//default, loc8_67, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.0fb: %AnyParam.as.Y.impl.K.type = import_ref PackageGenericInterface//default, loc9_22, loaded [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%PackageGenericInterface.import_ref.0fb), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .PackageGenericInterface = imports.%PackageGenericInterface
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %PackageGenericInterface.import = import PackageGenericInterface
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(imports.%PackageGenericInterface.import_ref.5ab: type) [from "has_generic_interface.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.55d)]
// CHECK:STDOUT:   %Self: @GenericInterface.%GenericInterface.type (%GenericInterface.type.55d) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.719)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageGenericInterface.import_ref.b37
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.0f3
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.f69
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: imports.%PackageGenericInterface.import_ref.321 as imports.%PackageGenericInterface.import_ref.ca6 [from "has_generic_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericInterface.import_ref.ba6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.601
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.3ad = binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.3ad = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.861 = var %obj.var_patt
// CHECK:STDOUT:   %.loc9_50.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_50.2: init %AnyParam.861 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8_3.1: init %AnyParam.861 = converted %.loc9_50.1, %.loc9_50.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc9_45: type = splice_block %AnyParam [concrete = constants.%AnyParam.861] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %PackageGenericInterface.ref: <namespace> = name_ref PackageGenericInterface, imports.%PackageGenericInterface [concrete = imports.%PackageGenericInterface]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.0da = name_ref GenericInterface, imports.%PackageGenericInterface.GenericInterface [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.861]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.861 = bind_name obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.861 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc10: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.f69 [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %impl.elem0: %.ed5 = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc10: %AnyParam.861 = bind_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method.loc10(%.loc10)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AnyParam.861, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%AnyParam.861, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %obj.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.596
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.596, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %obj.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.1ef = addr_of %obj.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.28c: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2f4)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.84b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.e57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic = %require_complete (constants.%require_complete.040)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K [from "has_generic_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.0da
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a64
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.2f4) {
// CHECK:STDOUT:   %Self => constants.%Self.2f4
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.84b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass: type = class_type @GenericClass, @GenericClass(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.17e: type = pattern_type %GenericClass.type [concrete]
// CHECK:STDOUT:   %AnyParam.0dd: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.2f4: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.84b: type = facet_access_type %Self.2f4 [symbolic]
// CHECK:STDOUT:   %pattern_type.e57: type = pattern_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %require_complete.040: <witness> = require_complete_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %AnyParam.0dd [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.0dd, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.0dd = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.494: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %.507: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AnyParam.0dd, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.846: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bde: %AggregateT.as_type.as.Destroy.impl.Op.type.846 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d96: type = ptr_type %AnyParam.0dd [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.bde, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst38 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.0f3 = import_ref PackageHasParam//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.f69: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %PackageHasParam.K: %Y.K.type = import_ref PackageHasParam//default, K, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.28c: %Y.type = import_ref PackageHasParam//default, inst44 [no loc], loaded [symbolic = constants.%Self.2f4]
// CHECK:STDOUT:   %PackageHasParam.import_ref.ce2: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericClass = %GenericClass.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericClass.decl: %GenericClass.type = class_decl @GenericClass [concrete = constants.%GenericClass.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc6_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_20.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @AnyParam.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.0dd]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_48: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (@AnyParam.as.Y.impl.%AnyParam.as.Y.impl.K.decl), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.0f3
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.f69
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.decl: %AnyParam.as.Y.impl.K.type = fn_decl @AnyParam.as.Y.impl.K [concrete = constants.%AnyParam.as.Y.impl.K] {
// CHECK:STDOUT:     %self.patt: %pattern_type.64f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.64f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.0dd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @AnyParam.as.Y.impl.%AnyParam [concrete = constants.%AnyParam.0dd]
// CHECK:STDOUT:     %self: %AnyParam.0dd = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %AnyParam.as.Y.impl.K.decl
// CHECK:STDOUT:   witness = file.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericClass(%U.loc6_20.2: type) {
// CHECK:STDOUT:   %U.loc6_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_20.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericClass
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.601
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.28c: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2f4)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.84b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.e57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic = %require_complete (constants.%require_complete.040)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K(%self.param: %AnyParam.0dd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.64f = binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.64f = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.0dd = var %obj.var_patt
// CHECK:STDOUT:   %.loc13_54.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_54.2: init %AnyParam.0dd = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3.1: init %AnyParam.0dd = converted %.loc13_54.1, %.loc13_54.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3.1
// CHECK:STDOUT:   %.loc13_49: type = splice_block %AnyParam [concrete = constants.%AnyParam.0dd] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.0dd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.0dd = bind_name obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.0dd = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.f69 [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %impl.elem0: %.507 = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.0dd = bind_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method.loc14(%.loc14)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AnyParam.0dd, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc13_3.2: %type_where = converted constants.%AnyParam.0dd, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %obj.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.bde
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.bde, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %obj.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d96 = addr_of %obj.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericClass(constants.%U) {
// CHECK:STDOUT:   %U.loc6_20.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) {
// CHECK:STDOUT:   %T => constants.%GenericClass.type
// CHECK:STDOUT:   %X => constants.%GenericClass.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.2f4) {
// CHECK:STDOUT:   %Self => constants.%Self.2f4
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.84b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.as_type => constants.%AnyParam.0dd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.64f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_class_as_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %GenericClass.type [concrete]
// CHECK:STDOUT:   %AnyParam.d71: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %pattern_type.312: type = pattern_type %AnyParam.d71 [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.d71 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.2f4: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.494: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.84b: type = facet_access_type %Self.2f4 [symbolic]
// CHECK:STDOUT:   %pattern_type.e57: type = pattern_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %require_complete.040: <witness> = require_complete_type %Self.as_type.84b [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness imports.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.d71, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %.79f: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AnyParam.d71, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.1a0: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.1aa: %AggregateT.as_type.as.Destroy.impl.Op.type.1a0 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8ee: type = ptr_type %AnyParam.d71 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.1aa, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageGenericClass: <namespace> = namespace file.%PackageGenericClass.import, [concrete] {
// CHECK:STDOUT:     .GenericClass = %PackageGenericClass.GenericClass
// CHECK:STDOUT:     import PackageGenericClass//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst38 [no loc], unloaded
// CHECK:STDOUT:   %PackageGenericClass.GenericClass: %GenericClass.type = import_ref PackageGenericClass//default, GenericClass, loaded [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.5ab: type = import_ref PackageGenericClass//default, loc6_20, loaded [symbolic = @GenericClass.%U (constants.%U)]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.8f2: <witness> = import_ref PackageGenericClass//default, loc6_31, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.065 = import_ref PackageGenericClass//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.0f3 = import_ref PackageHasParam//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.f69: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.ce2: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.28c: %Y.type = import_ref PackageHasParam//default, inst44 [no loc], loaded [symbolic = constants.%Self.2f4]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.812: <witness> = import_ref PackageGenericClass//default, loc8_66, loaded [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.a0e: type = import_ref PackageGenericClass//default, loc8_43, loaded [concrete = constants.%AnyParam.d71]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.ca6: type = import_ref PackageGenericClass//default, loc8_63, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.434: %AnyParam.as.Y.impl.K.type = import_ref PackageGenericClass//default, loc9_22, loaded [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%PackageGenericClass.import_ref.434), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .PackageGenericClass = imports.%PackageGenericClass
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %PackageGenericClass.import = import PackageGenericClass
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.0f3
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.f69
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: imports.%PackageGenericClass.import_ref.a0e as imports.%PackageGenericClass.import_ref.ca6 [from "has_generic_class.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericClass.import_ref.812
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.601
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericClass(imports.%PackageGenericClass.import_ref.5ab: type) [from "has_generic_class.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageGenericClass.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageGenericClass.import_ref.065
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.312 = binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.312 = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.d71 = var %obj.var_patt
// CHECK:STDOUT:   %.loc8_74.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_74.2: init %AnyParam.d71 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8_3.1: init %AnyParam.d71 = converted %.loc8_74.1, %.loc8_74.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_69: type = splice_block %AnyParam [concrete = constants.%AnyParam.d71] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %PackageGenericClass.ref: <namespace> = name_ref PackageGenericClass, imports.%PackageGenericClass [concrete = imports.%PackageGenericClass]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, imports.%PackageGenericClass.GenericClass [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.d71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.d71 = bind_name obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.d71 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc9: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.f69 [concrete = constants.%assoc0.494]
// CHECK:STDOUT:   %impl.elem0: %.79f = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc9: %AnyParam.d71 = bind_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method.loc9(%.loc9)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AnyParam.d71, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%AnyParam.d71, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %obj.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.1aa
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.1aa, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %obj.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.8ee = addr_of %obj.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.28c: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2f4)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.84b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.e57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic = %require_complete (constants.%require_complete.040)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K [from "has_generic_class.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericClass(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) {
// CHECK:STDOUT:   %T => constants.%GenericClass.type
// CHECK:STDOUT:   %X => constants.%GenericClass.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.2f4) {
// CHECK:STDOUT:   %Self => constants.%Self.2f4
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.84b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %Self.d36: %Extra1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Self.a47: %Extra2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Self.c02: %Extra3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Self.696: %Extra4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Self.b07: %Extra5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Self.5c5: %Extra6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Self.f52: %Extra7.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Self.779: %Extra8.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.7ee [symbolic]
// CHECK:STDOUT:   %pattern_type.d22: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %tuple.type.c53: type = tuple_type (type, type, type, type, type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.f7c: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.763: type = class_type @C, @C(%tuple.type.f7c) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.fc9: type = pattern_type %C.763 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.763, (%I.impl_witness) [concrete]
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
// CHECK:STDOUT:     .Extra1 = %Extra1.decl
// CHECK:STDOUT:     .Extra2 = %Extra2.decl
// CHECK:STDOUT:     .Extra3 = %Extra3.decl
// CHECK:STDOUT:     .Extra4 = %Extra4.decl
// CHECK:STDOUT:     .Extra5 = %Extra5.decl
// CHECK:STDOUT:     .Extra6 = %Extra6.decl
// CHECK:STDOUT:     .Extra7 = %Extra7.decl
// CHECK:STDOUT:     .Extra8 = %Extra8.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Extra1.decl: type = interface_decl @Extra1 [concrete = constants.%Extra1.type] {} {}
// CHECK:STDOUT:   %Extra2.decl: type = interface_decl @Extra2 [concrete = constants.%Extra2.type] {} {}
// CHECK:STDOUT:   %Extra3.decl: type = interface_decl @Extra3 [concrete = constants.%Extra3.type] {} {}
// CHECK:STDOUT:   %Extra4.decl: type = interface_decl @Extra4 [concrete = constants.%Extra4.type] {} {}
// CHECK:STDOUT:   %Extra5.decl: type = interface_decl @Extra5 [concrete = constants.%Extra5.type] {} {}
// CHECK:STDOUT:   %Extra6.decl: type = interface_decl @Extra6 [concrete = constants.%Extra6.type] {} {}
// CHECK:STDOUT:   %Extra7.decl: type = interface_decl @Extra7 [concrete = constants.%Extra7.type] {} {}
// CHECK:STDOUT:   %Extra8.decl: type = interface_decl @Extra8 [concrete = constants.%Extra8.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc13_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Extra1.ref: type = name_ref Extra1, file.%Extra1.decl [concrete = constants.%Extra1.type]
// CHECK:STDOUT:     %Extra2.ref: type = name_ref Extra2, file.%Extra2.decl [concrete = constants.%Extra2.type]
// CHECK:STDOUT:     %Extra3.ref: type = name_ref Extra3, file.%Extra3.decl [concrete = constants.%Extra3.type]
// CHECK:STDOUT:     %Extra4.ref: type = name_ref Extra4, file.%Extra4.decl [concrete = constants.%Extra4.type]
// CHECK:STDOUT:     %Extra5.ref: type = name_ref Extra5, file.%Extra5.decl [concrete = constants.%Extra5.type]
// CHECK:STDOUT:     %Extra6.ref: type = name_ref Extra6, file.%Extra6.decl [concrete = constants.%Extra6.type]
// CHECK:STDOUT:     %Extra7.ref: type = name_ref Extra7, file.%Extra7.decl [concrete = constants.%Extra7.type]
// CHECK:STDOUT:     %Extra8.ref: type = name_ref Extra8, file.%Extra8.decl [concrete = constants.%Extra8.type]
// CHECK:STDOUT:     %.loc16_71: %tuple.type.c53 = tuple_literal (%Extra1.ref, %Extra2.ref, %Extra3.ref, %Extra4.ref, %Extra5.ref, %Extra6.ref, %Extra7.ref, %Extra8.ref)
// CHECK:STDOUT:     %.loc16_72: type = converted %.loc16_71, constants.%tuple.type.f7c [concrete = constants.%tuple.type.f7c]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.type.f7c) [concrete = constants.%C.763]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@C.as.I.impl.%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT:   %Self: %Extra1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d36]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT:   %Self: %Extra2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a47]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT:   %Self: %Extra3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.c02]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT:   %Self: %Extra4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.696]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT:   %Self: %Extra5.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b07]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT:   %Self: %Extra6.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.5c5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT:   %Self: %Extra7.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f52]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT:   %Self: %Extra8.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.779]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %self.patt: @I.F.%pattern_type (%pattern_type.d22) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.F.%pattern_type (%pattern_type.d22) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.F.%Self.as_type.loc14_26.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.2 [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:       %Self.as_type.loc14_26.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc14_26.2: type = converted %Self.ref, %Self.as_type.loc14_26.2 [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.F.%Self.as_type.loc14_26.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fc9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fc9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.763 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C [concrete = constants.%C.763]
// CHECK:STDOUT:     %self: %C.763 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc13_9.2: type) {
// CHECK:STDOUT:   %T.loc13_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.7ee)]
// CHECK:STDOUT:   %Self.as_type.loc14_26.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc14_26.1 [symbolic = %pattern_type (constants.%pattern_type.d22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.as_type.loc14_26.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C.763) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc13_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.7ee) {
// CHECK:STDOUT:   %Self => constants.%Self.7ee
// CHECK:STDOUT:   %Self.as_type.loc14_26.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type.f7c) {
// CHECK:STDOUT:   %T.loc13_9.1 => constants.%tuple.type.f7c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc14_26.1 => constants.%C.763
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.42e: type = class_type @C, @C(type) [concrete]
// CHECK:STDOUT:   %pattern_type.6e4: type = pattern_type %C.42e [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.caa: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%HasExtraInterfaces.import_ref.d54 [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.caa [symbolic]
// CHECK:STDOUT:   %pattern_type.95f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.c07: type = class_type @C, @C(%tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces: <namespace> = namespace file.%HasExtraInterfaces.import, [concrete] {
// CHECK:STDOUT:     .C = %HasExtraInterfaces.C
// CHECK:STDOUT:     .I = %HasExtraInterfaces.I
// CHECK:STDOUT:     import HasExtraInterfaces//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces.C: %C.type = import_ref HasExtraInterfaces//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.5ab: type = import_ref HasExtraInterfaces//default, loc13_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.8f2: <witness> = import_ref HasExtraInterfaces//default, loc13_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.4c0 = import_ref HasExtraInterfaces//default, inst69 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.I: type = import_ref HasExtraInterfaces//default, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.8e7 = import_ref HasExtraInterfaces//default, inst75 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.be9: %I.assoc_type = import_ref HasExtraInterfaces//default, loc14_33, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %HasExtraInterfaces.F = import_ref HasExtraInterfaces//default, F, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.d54: %I.F.type = import_ref HasExtraInterfaces//default, loc14_33, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.59a: %I.type = import_ref HasExtraInterfaces//default, inst75 [no loc], loaded [symbolic = constants.%Self.caa]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.80c = import_ref HasExtraInterfaces//default, loc16_79, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.23a = import_ref HasExtraInterfaces//default, inst54 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.a8e = import_ref HasExtraInterfaces//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.8be = import_ref HasExtraInterfaces//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.04c = import_ref HasExtraInterfaces//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.29b = import_ref HasExtraInterfaces//default, inst34 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.330 = import_ref HasExtraInterfaces//default, inst29 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.636 = import_ref HasExtraInterfaces//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.d01 = import_ref HasExtraInterfaces//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.ef7: type = import_ref HasExtraInterfaces//default, loc16_72, loaded [concrete = constants.%C.c07]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.301: type = import_ref HasExtraInterfaces//default, loc16_77, loaded [concrete = constants.%I.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasExtraInterfaces = imports.%HasExtraInterfaces
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasExtraInterfaces.import = import HasExtraInterfaces
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %c.patt: %pattern_type.6e4 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6e4 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.42e = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %C [concrete = constants.%C.42e] {
// CHECK:STDOUT:       %HasExtraInterfaces.ref.loc5: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [concrete = imports.%HasExtraInterfaces]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%HasExtraInterfaces.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type) [concrete = constants.%C.42e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.42e = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.8e7
// CHECK:STDOUT:   .F = imports.%HasExtraInterfaces.import_ref.be9
// CHECK:STDOUT:   witness = (imports.%HasExtraInterfaces.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.23a
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.a8e
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.8be
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.04c
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.29b
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.330
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.636
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.d01
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%HasExtraInterfaces.import_ref.ef7 as imports.%HasExtraInterfaces.import_ref.301 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%HasExtraInterfaces.import_ref.80c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%HasExtraInterfaces.import_ref.5ab: type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%HasExtraInterfaces.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%HasExtraInterfaces.import_ref.4c0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%c.param: %C.42e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.42e = name_ref c, %c
// CHECK:STDOUT:   %HasExtraInterfaces.ref.loc12: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [concrete = imports.%HasExtraInterfaces]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%HasExtraInterfaces.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%HasExtraInterfaces.import_ref.be9 [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%HasExtraInterfaces.import_ref.59a: %I.type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.caa)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.95f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.caa) {
// CHECK:STDOUT:   %Self => constants.%Self.caa
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.95f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type) {
// CHECK:STDOUT:   %T => constants.%tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
