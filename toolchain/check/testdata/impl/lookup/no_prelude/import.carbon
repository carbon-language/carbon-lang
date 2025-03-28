// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/no_prelude/import.carbon

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
// CHECK:STDOUT:   %F.type.b7b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.f50: %F.type.b7b = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type %HasF.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.a02: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.dc7: %F.type.a02 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.b7b = fn_decl @F.1 [concrete = constants.%F.f50] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.a02 = fn_decl @F.2 [concrete = constants.%F.dc7] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.d42: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.133: type = facet_access_type %Self.d42 [symbolic]
// CHECK:STDOUT:   %G.type.d27: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.688: %G.type.d27 = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type %HasG.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, @HasG.%G.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %impl_witness.722: <witness> = impl_witness (@impl.e95.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.05c: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.486: %G.type.05c = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.062: %HasG.type = facet_value %C, (%impl_witness.722) [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.cf3: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.dbc: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.a2b: %F.type.dbc = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.7cf: type = facet_access_type %Self.cf3 [symbolic]
// CHECK:STDOUT:   %impl_witness.a36: <witness> = impl_witness (@impl.8af.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.bab: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.46f: %F.type.bab = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%impl_witness.a36) [concrete]
// CHECK:STDOUT:   %impl_witness.9ed: <witness> = impl_witness (@impl.da9.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.c1d: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.294: %G.type.c1d = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.548: %HasG.type = facet_value %D, (%impl_witness.9ed) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .C = %PackageA.C
// CHECK:STDOUT:     .HasF = %PackageA.HasF
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.C: type = import_ref PackageA//default, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.28c = import_ref PackageA//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.a2a = import_ref PackageA//default, loc5_21, unloaded
// CHECK:STDOUT:   %PackageA.F: %F.type.dbc = import_ref PackageA//default, F, loaded [concrete = constants.%F.a2b]
// CHECK:STDOUT:   %PackageA.import_ref.e73: %HasF.type = import_ref PackageA//default, inst16 [no loc], loaded [symbolic = constants.%Self.cf3]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [concrete = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.e95 [concrete] {} {
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.e95.%G.decl) [concrete = constants.%impl_witness.722]
// CHECK:STDOUT:   impl_decl @impl.8af [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.8af.%F.decl) [concrete = constants.%impl_witness.a36]
// CHECK:STDOUT:   impl_decl @impl.da9 [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.da9.%G.decl) [concrete = constants.%impl_witness.9ed]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d42]
// CHECK:STDOUT:   %G.decl: %G.type.d27 = fn_decl @G.1 [concrete = constants.%G.688] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc7_14.1 (%Self.as_type.133) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc7_14.1 (%Self.as_type.133) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc7_14.1 (%Self.as_type.133) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.133)] {
// CHECK:STDOUT:       %Self.ref: %HasG.type = name_ref Self, @HasG.%Self [symbolic = %Self (constants.%Self.d42)]
// CHECK:STDOUT:       %Self.as_type.loc7_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.133)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.133)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc7_14.1 (%Self.as_type.133) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.28c
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.a2a
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.e95: %C.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.05c = fn_decl @G.2 [concrete = constants.%G.486] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.e95.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8af: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.bab = fn_decl @F.2 [concrete = constants.%F.46f] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %D = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.8af.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.da9: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.c1d = fn_decl @G.3 [concrete = constants.%G.294] {
// CHECK:STDOUT:     %self.patt: %D = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %D = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.da9.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @G.1(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.d42)]
// CHECK:STDOUT:   %Self.as_type.loc7_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type.133)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc7_14.1 (%Self.as_type.133)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %C]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%PackageA.import_ref.e73: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf3)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.7cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type.7cf)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %D]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3[%self.param_patt: %D]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.d42) {
// CHECK:STDOUT:   %Self => constants.%Self.d42
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%Self.as_type.133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet.062) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.062
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.cf3) {
// CHECK:STDOUT:   %Self => constants.%Self.cf3
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.7cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet.548) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.548
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type %HasF.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.ab2 [concrete]
// CHECK:STDOUT:   %F.type.dbc: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageA.import_ref.148) [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.a6c: type = fn_type_with_self_type %F.type.dbc, %HasF.facet [concrete]
// CHECK:STDOUT:   %F.type.4e3: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.857: %F.type.4e3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [concrete] {
// CHECK:STDOUT:     .C = %PackageA.C
// CHECK:STDOUT:     .HasF = %PackageA.HasF
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.C: type = import_ref PackageA//default, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.28c = import_ref PackageA//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.566: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.e73: %HasF.type = import_ref PackageA//default, inst16 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageA.import_ref.a71: <witness> = import_ref PackageA//default, loc11_16, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageA.import_ref.29a: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.e8c: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .TestCF = %TestCF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %TestCF.decl: %TestCF.type = fn_decl @TestCF [concrete = constants.%TestCF] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
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
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.28c
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.566
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%PackageA.import_ref.29a as imports.%PackageA.import_ref.e8c [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.a71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc7: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.566 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.a6c = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%F.857]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%PackageA.import_ref.e73: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C]() [from "package_a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_df.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.cf3: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type %HasF.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.ab2 [concrete]
// CHECK:STDOUT:   %F.type.dbc: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.cf3 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageB.import_ref.0cd) [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.769: type = fn_type_with_self_type %F.type.dbc, %HasF.facet [concrete]
// CHECK:STDOUT:   %F.type.394: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.1fc: %F.type.394 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
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
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.28c = import_ref PackageA//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.566: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.e73: %HasF.type = import_ref PackageA//default, inst16 [no loc], loaded [symbolic = constants.%Self.cf3]
// CHECK:STDOUT:   %PackageA.import_ref.0e8 = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.29a: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.e8c: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.5d8 = import_ref PackageB//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.ed7 = import_ref PackageB//default, loc7_21, unloaded
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.fa0 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.dfb: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.f2f: <witness> = import_ref PackageB//default, loc18_25, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.1: type = import_ref PackageB//default, loc18_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.831: type = import_ref PackageB//default, loc18_19, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.231 = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.2: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDF = %TestDF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDF.decl: %TestDF.type = fn_decl @TestDF [concrete = constants.%TestDF] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, call_param0
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
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.28c
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.566
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.5d8
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.ed7
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4b2: imports.%PackageA.import_ref.29a as imports.%PackageA.import_ref.e8c [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.0e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.922: imports.%PackageB.import_ref.dfb as imports.%PackageB.import_ref.cee586.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.52b: imports.%PackageB.import_ref.aa9f8a.1 as imports.%PackageB.import_ref.831 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.47e: imports.%PackageB.import_ref.aa9f8a.2 as imports.%PackageB.import_ref.cee586.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.231
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
// CHECK:STDOUT: fn @TestDF(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.566 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.769 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%F.1fc]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%PackageA.import_ref.e73: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.cf3)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %D]() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.cf3) {
// CHECK:STDOUT:   %Self => constants.%Self.cf3
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.fcb: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type %HasG.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.70a [concrete]
// CHECK:STDOUT:   %G.type.d9e: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.fcb [symbolic]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageB.import_ref.9ec) [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.145: type = fn_type_with_self_type %G.type.d9e, %HasG.facet [concrete]
// CHECK:STDOUT:   %G.type.18e: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.dbb: %G.type.18e = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
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
// CHECK:STDOUT:   %PackageA.import_ref.2c4 = import_ref PackageA//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.5d8 = import_ref PackageB//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.604: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.ef5: %HasG.type = import_ref PackageB//default, inst18 [no loc], loaded [symbolic = constants.%Self.fcb]
// CHECK:STDOUT:   %PackageA.import_ref.28c = import_ref PackageA//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.a2a = import_ref PackageA//default, loc5_21, unloaded
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.0e8 = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.29a: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.e8c: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.5d9: <witness> = import_ref PackageB//default, loc13_25, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.dfb: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.7db = import_ref PackageB//default, loc18_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.1: type = import_ref PackageB//default, loc18_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.831: type = import_ref PackageB//default, loc18_19, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.231 = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.2: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestCG = %TestCG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestCG.decl: %TestCG.type = fn_decl @TestCG [concrete = constants.%TestCG] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
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
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.5d8
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.604
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.28c
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.a2a
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4b2: imports.%PackageA.import_ref.29a as imports.%PackageA.import_ref.e8c [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.0e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.922: imports.%PackageB.import_ref.dfb as imports.%PackageB.import_ref.cee586.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.5d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.52b: imports.%PackageB.import_ref.aa9f8a.1 as imports.%PackageB.import_ref.831 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.7db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.47e: imports.%PackageB.import_ref.aa9f8a.2 as imports.%PackageB.import_ref.cee586.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.231
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
// CHECK:STDOUT: fn @TestCG(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.604 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.145 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G.dbb]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(imports.%PackageB.import_ref.ef5: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.fcb)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %C]() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.fcb) {
// CHECK:STDOUT:   %Self => constants.%Self.fcb
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_dg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.fcb: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type %HasG.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.70a [concrete]
// CHECK:STDOUT:   %G.type.d9e: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.fcb [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageB.import_ref.b0a) [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %D, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %G.type.d9e, %HasG.facet [concrete]
// CHECK:STDOUT:   %G.type.405: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.703: %G.type.405 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [concrete] {
// CHECK:STDOUT:     .D = %PackageB.D
// CHECK:STDOUT:     .HasG = %PackageB.HasG
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB.D: type = import_ref PackageB//default, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.cab = import_ref PackageB//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.5d8 = import_ref PackageB//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.604: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.ef5: %HasG.type = import_ref PackageB//default, inst18 [no loc], loaded [symbolic = constants.%Self.fcb]
// CHECK:STDOUT:   %PackageB.import_ref.fa0 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8db: <witness> = import_ref PackageB//default, inst47 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.6a9 = import_ref PackageB//default, inst48 [indirect], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.dfb: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.96f = import_ref PackageB//default, inst71 [indirect], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.b30 = import_ref PackageB//default, inst72 [indirect], unloaded
// CHECK:STDOUT:   %PackageB.F = import_ref PackageB//default, F, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.7db = import_ref PackageB//default, loc18_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.1: type = import_ref PackageB//default, loc18_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.831: type = import_ref PackageB//default, loc18_19, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.240: <witness> = import_ref PackageB//default, loc23_16, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.aa9f8a.2: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.cee586.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDG = %TestDG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDG.decl: %TestDG.type = fn_decl @TestDG [concrete = constants.%TestDG] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, call_param0
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
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.5d8
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.604
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.96f
// CHECK:STDOUT:   .F = imports.%PackageB.import_ref.b30
// CHECK:STDOUT:   witness = (imports.%PackageB.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.162: imports.%PackageB.import_ref.dfb as imports.%PackageB.import_ref.cee586.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5f1: imports.%PackageB.import_ref.aa9f8a.1 as imports.%PackageB.import_ref.831 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.7db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.47e: imports.%PackageB.import_ref.aa9f8a.2 as imports.%PackageB.import_ref.cee586.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.240
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
// CHECK:STDOUT: fn @TestDG(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc7: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.604 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.b47 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G.703]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(imports.%PackageB.import_ref.ef5: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.fcb)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %D]() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.fcb) {
// CHECK:STDOUT:   %Self => constants.%Self.fcb
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %H.type.474: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %H.c1d: %H.type.474 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type %Z.type [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%H.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%H.decl) [concrete]
// CHECK:STDOUT:   %H.type.707: type = fn_type @H.2 [concrete]
// CHECK:STDOUT:   %H.8d7: %H.type.707 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%H.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %H.decl: %H.type.474 = fn_decl @H.1 [concrete = constants.%H.c1d] {
// CHECK:STDOUT:     %self.patt: @H.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @H.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @H.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @H.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %H.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %assoc0
// CHECK:STDOUT:   witness = (%H.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %H.decl: %H.type.707 = fn_decl @H.2 [concrete = constants.%H.8d7] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @H.1.%Self.as_type.loc5_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2[%self.param_patt: %empty_tuple.type]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%empty_tuple.type
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
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type %Z.type [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, imports.%PackageAssociatedInterface.import_ref.250 [concrete]
// CHECK:STDOUT:   %H.type.386: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageAssociatedInterface.import_ref.6d7) [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.c44: type = fn_type_with_self_type %H.type.386, %Z.facet [concrete]
// CHECK:STDOUT:   %H.type.ab3: type = fn_type @H.2 [concrete]
// CHECK:STDOUT:   %H.c25: %H.type.ab3 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageAssociatedInterface: <namespace> = namespace file.%PackageAssociatedInterface.import, [concrete] {
// CHECK:STDOUT:     .Z = %PackageAssociatedInterface.Z
// CHECK:STDOUT:     import PackageAssociatedInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageAssociatedInterface.Z: type = import_ref PackageAssociatedInterface//default, Z, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.f88 = import_ref PackageAssociatedInterface//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.ddc: %Z.assoc_type = import_ref PackageAssociatedInterface//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageAssociatedInterface.H = import_ref PackageAssociatedInterface//default, H, unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.d26: %Z.type = import_ref PackageAssociatedInterface//default, inst16 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.998: <witness> = import_ref PackageAssociatedInterface//default, loc8_14, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.e5c: type = import_ref PackageAssociatedInterface//default, loc8_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.df1: type = import_ref PackageAssociatedInterface//default, loc8_12, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageAssociatedInterface = imports.%PackageAssociatedInterface
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageAssociatedInterface.import = import PackageAssociatedInterface
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageAssociatedInterface.import_ref.f88
// CHECK:STDOUT:   .H = imports.%PackageAssociatedInterface.import_ref.ddc
// CHECK:STDOUT:   witness = (imports.%PackageAssociatedInterface.H)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%PackageAssociatedInterface.import_ref.e5c as imports.%PackageAssociatedInterface.import_ref.df1 [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageAssociatedInterface.import_ref.998
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_4.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %PackageAssociatedInterface.ref: <namespace> = name_ref PackageAssociatedInterface, imports.%PackageAssociatedInterface [concrete = imports.%PackageAssociatedInterface]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, imports.%PackageAssociatedInterface.Z [concrete = constants.%Z.type]
// CHECK:STDOUT:   %H.ref: %Z.assoc_type = name_ref H, imports.%PackageAssociatedInterface.import_ref.ddc [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.c44 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%H.c25]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc7_4.1, %impl.elem0
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_4.2: %empty_tuple.type = converted %.loc7_4.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %bound_method(%.loc7_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H.1(imports.%PackageAssociatedInterface.import_ref.d26: %Z.type) [from "associated_interface.carbon"] {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @H.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2[%self.param_patt: %empty_tuple.type]() [from "associated_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type %Y.type [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%K.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AnyParam = %AnyParam.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AnyParam.decl: %AnyParam.type = class_decl @AnyParam [concrete = constants.%AnyParam.generic] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc4_26.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [concrete = constants.%K] {
// CHECK:STDOUT:     %self.patt: @K.%Self.as_type.loc7_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @K.%Self.as_type.loc7_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @K.%Self.as_type.loc7_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Y.type = name_ref Self, @Y.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc7_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @K.%Self.as_type.loc7_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, %K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(%T.loc4_16.1: type, %X.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T)) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %X.loc4_26.2: @AnyParam.%T.loc4_16.2 (%T) = bind_symbolic_name X, 1 [symbolic = %X.loc4_26.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc4_26.2: @AnyParam.%T.loc4_16.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%AnyParam
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K(@Y.%Self: %Y.type) {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc7_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @K.%Self.as_type.loc7_14.1 (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @K.%Self.as_type.loc7_14.1 (%Self.as_type)]() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt
// CHECK:STDOUT:   %X.loc4_26.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc4_26.2 => constants.%X.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.c92: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.c92 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericInterface.type.3fe: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.a1c: %GenericInterface.type.3fe = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.51c: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.e4e: %GenericInterface.type.c92 = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.241: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.c92, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.f64: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.311: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %K.7a1: %K.type.311 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.f64 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete]
// CHECK:STDOUT:   %K.type.dcd: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.2e9: %K.type.dcd = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.241, (%impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.241 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type %Y.type [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %.bae: type = fn_type_with_self_type %K.type.311, %Y.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.dc1 = import_ref PackageHasParam//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.5e7: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageHasParam.K: %K.type.311 = import_ref PackageHasParam//default, K, loaded [concrete = constants.%K.7a1]
// CHECK:STDOUT:   %PackageHasParam.import_ref.292: %Y.type = import_ref PackageHasParam//default, inst37 [no loc], loaded [symbolic = constants.%Self.f64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.c92 = interface_decl @GenericInterface [concrete = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %U.patt.loc6_28.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc6_28.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.c92 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.241]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_52: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%U.loc6_28.1: type) {
// CHECK:STDOUT:   %U.loc6_28.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_28.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U.loc6_28.2)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.3fe)]
// CHECK:STDOUT:   %Self.2: @GenericInterface.%GenericInterface.type (%GenericInterface.type.3fe) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a1c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.3fe) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a1c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.dc1
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.5e7
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %K.decl: %K.type.dcd = fn_decl @K.2 [concrete = constants.%K.2e9] {
// CHECK:STDOUT:     %self.patt: %AnyParam.241 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AnyParam.241 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.241 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%AnyParam [concrete = constants.%AnyParam.241]
// CHECK:STDOUT:     %self: %AnyParam.241 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.51c)]
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
// CHECK:STDOUT: generic fn @K.1(imports.%PackageHasParam.import_ref.292: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @K.1.%Self.as_type (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @K.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2[%self.param_patt: %AnyParam.241]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %AnyParam.241 = binding_pattern obj
// CHECK:STDOUT:     %.loc13_3.1: %AnyParam.241 = var_pattern %obj.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.241 = var obj
// CHECK:STDOUT:   %.loc13_58.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_58.2: init %AnyParam.241 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3.2: init %AnyParam.241 = converted %.loc13_58.1, %.loc13_58.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3.2
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
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.5e7 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.bae = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%K.2e9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.241 = bind_value %obj.ref
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %bound_method(%.loc14)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_28.2 => constants.%U.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%U.loc6_28.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X.patt.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.c92
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %X.patt => constants.%X.patt.e4e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.as_type => constants.%AnyParam.241
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.51c: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.0da: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.138: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.13b: %GenericInterface.type.138 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.978: %GenericInterface.type.0da = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.861: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.0da, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.861 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.f64: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type %Y.type [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %K.type.311: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.f64 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageGenericInterface.import_ref.456) [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.861, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.808: type = fn_type_with_self_type %K.type.311, %Y.facet [concrete]
// CHECK:STDOUT:   %K.type.7f9: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.c3c: %K.type.7f9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %PackageGenericInterface.GenericInterface: %GenericInterface.type.0da = import_ref PackageGenericInterface//default, GenericInterface, loaded [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.5ab: type = import_ref PackageGenericInterface//default, loc6_28, loaded [symbolic = @GenericInterface.%U (constants.%U)]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.c3b = import_ref PackageGenericInterface//default, inst27 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.dc1 = import_ref PackageHasParam//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.5e7: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.292: %Y.type = import_ref PackageHasParam//default, inst37 [no loc], loaded [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.ca8: <witness> = import_ref PackageGenericInterface//default, loc8_70, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.321: type = import_ref PackageGenericInterface//default, loc8_47, loaded [concrete = constants.%AnyParam.861]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.ca6: type = import_ref PackageGenericInterface//default, loc8_67, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .PackageGenericInterface = imports.%PackageGenericInterface
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %PackageGenericInterface.import = import PackageGenericInterface
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(imports.%PackageGenericInterface.import_ref.5ab: type) [from "has_generic_interface.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.138)]
// CHECK:STDOUT:   %Self: @GenericInterface.%GenericInterface.type (%GenericInterface.type.138) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.13b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageGenericInterface.import_ref.c3b
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.dc1
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.5e7
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%PackageGenericInterface.import_ref.321 as imports.%PackageGenericInterface.import_ref.ca6 [from "has_generic_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericInterface.import_ref.ca8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.51c)]
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
// CHECK:STDOUT:     %obj.patt: %AnyParam.861 = binding_pattern obj
// CHECK:STDOUT:     %.loc8_3.1: %AnyParam.861 = var_pattern %obj.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.861 = var obj
// CHECK:STDOUT:   %.loc9_50.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_50.2: init %AnyParam.861 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8_3.2: init %AnyParam.861 = converted %.loc9_50.1, %.loc9_50.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8_3.2
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
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.5e7 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.808 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%K.c3c]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc10: %AnyParam.861 = bind_value %obj.ref
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %bound_method(%.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.1(imports.%PackageHasParam.import_ref.292: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @K.1.%Self.as_type (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @K.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2[%self.param_patt: %AnyParam.861]() [from "has_generic_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X.patt.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.0da
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %X.patt => constants.%X.patt.978
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass: type = class_type @GenericClass, @GenericClass(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.51c: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.2a8: %GenericClass.type = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.0dd: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.311: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %K.7a1: %K.type.311 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete]
// CHECK:STDOUT:   %K.type.965: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.5a9: %K.type.965 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.0dd, (%impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.0dd = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type %Y.type [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %.e27: type = fn_type_with_self_type %K.type.311, %Y.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [concrete] {
// CHECK:STDOUT:     .AnyParam = %PackageHasParam.AnyParam
// CHECK:STDOUT:     .Y = %PackageHasParam.Y
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.AnyParam: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:   %PackageHasParam.import_ref.5ab: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.34c: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.dc1 = import_ref PackageHasParam//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.5e7: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageHasParam.K: %K.type.311 = import_ref PackageHasParam//default, K, loaded [concrete = constants.%K.7a1]
// CHECK:STDOUT:   %PackageHasParam.import_ref.292: %Y.type = import_ref PackageHasParam//default, inst37 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericClass = %GenericClass.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericClass.decl: %GenericClass.type = class_decl @GenericClass [concrete = constants.%GenericClass.generic] {
// CHECK:STDOUT:     %U.patt.loc6_20.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc6_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.0dd]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_48: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.dc1
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.5e7
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %K.decl: %K.type.965 = fn_decl @K.2 [concrete = constants.%K.5a9] {
// CHECK:STDOUT:     %self.patt: %AnyParam.0dd = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AnyParam.0dd = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.0dd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%AnyParam [concrete = constants.%AnyParam.0dd]
// CHECK:STDOUT:     %self: %AnyParam.0dd = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericClass(%U.loc6_20.1: type) {
// CHECK:STDOUT:   %U.loc6_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_20.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_20.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_20.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericClass
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.51c)]
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
// CHECK:STDOUT: generic fn @K.1(imports.%PackageHasParam.import_ref.292: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @K.1.%Self.as_type (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @K.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2[%self.param_patt: %AnyParam.0dd]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %AnyParam.0dd = binding_pattern obj
// CHECK:STDOUT:     %.loc13_3.1: %AnyParam.0dd = var_pattern %obj.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.0dd = var obj
// CHECK:STDOUT:   %.loc13_54.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_54.2: init %AnyParam.0dd = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3.2: init %AnyParam.0dd = converted %.loc13_54.1, %.loc13_54.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3.2
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
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.5e7 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.e27 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%K.5a9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.0dd = bind_value %obj.ref
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %bound_method(%.loc14)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericClass(constants.%U) {
// CHECK:STDOUT:   %U.loc6_20.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_20.2 => constants.%U.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X.patt.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) {
// CHECK:STDOUT:   %T => constants.%GenericClass.type
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%GenericClass.generic
// CHECK:STDOUT:   %X.patt => constants.%X.patt.2a8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.as_type => constants.%AnyParam.0dd
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.51c: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.7fb: %GenericClass.type = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.d71: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.d71 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type %Y.type [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.ce2 [concrete]
// CHECK:STDOUT:   %K.type.311: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%PackageGenericClass.import_ref.fbf) [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.d71, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.3b2: type = fn_type_with_self_type %K.type.311, %Y.facet [concrete]
// CHECK:STDOUT:   %K.type.9cf: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.039: %K.type.9cf = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.601 = import_ref PackageHasParam//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %PackageGenericClass.GenericClass: %GenericClass.type = import_ref PackageGenericClass//default, GenericClass, loaded [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.5ab: type = import_ref PackageGenericClass//default, loc6_20, loaded [symbolic = @GenericClass.%U (constants.%U)]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.8f2: <witness> = import_ref PackageGenericClass//default, loc6_31, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.065 = import_ref PackageGenericClass//default, inst26 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.dc1 = import_ref PackageHasParam//default, inst37 [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.5e7: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.292: %Y.type = import_ref PackageHasParam//default, inst37 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.f23: <witness> = import_ref PackageGenericClass//default, loc8_66, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.a0e: type = import_ref PackageGenericClass//default, loc8_43, loaded [concrete = constants.%AnyParam.d71]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.ca6: type = import_ref PackageGenericClass//default, loc8_63, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .PackageGenericClass = imports.%PackageGenericClass
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %PackageGenericClass.import = import PackageGenericClass
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.dc1
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.5e7
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%PackageGenericClass.import_ref.a0e as imports.%PackageGenericClass.import_ref.ca6 [from "has_generic_class.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericClass.import_ref.f23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.5ab: type, imports.%PackageHasParam.import_ref.34c: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.51c)]
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
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
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
// CHECK:STDOUT:     %obj.patt: %AnyParam.d71 = binding_pattern obj
// CHECK:STDOUT:     %.loc8_3.1: %AnyParam.d71 = var_pattern %obj.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.d71 = var obj
// CHECK:STDOUT:   %.loc8_74.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_74.2: init %AnyParam.d71 = class_init (), %obj.var [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8_3.2: init %AnyParam.d71 = converted %.loc8_74.1, %.loc8_74.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8_3.2
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
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.5e7 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.3b2 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%K.039]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc9: %AnyParam.d71 = bind_value %obj.ref
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %bound_method(%.loc9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.1(imports.%PackageHasParam.import_ref.292: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @K.1.%Self.as_type (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @K.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2[%self.param_patt: %AnyParam.d71]() [from "has_generic_class.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X.patt.51c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericClass(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) {
// CHECK:STDOUT:   %T => constants.%GenericClass.type
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %X => constants.%GenericClass.generic
// CHECK:STDOUT:   %X.patt => constants.%X.patt.7fb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %Self.66c: %Extra1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Self.2ed: %Extra2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Self.622: %Extra3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Self.234: %Extra4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Self.d1b: %Extra5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Self.aeb: %Extra6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Self.b7e: %Extra7.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Self.f90: %Extra8.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %tuple.type.c53: type = tuple_type (type, type, type, type, type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.15d: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.69b: type = class_type @C, @C(%tuple.type.15d) [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.06f: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.13c: %F.type.06f = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.69b, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT:   %Extra1.decl: type = interface_decl @Extra1 [concrete = constants.%Extra1.type] {} {}
// CHECK:STDOUT:   %Extra2.decl: type = interface_decl @Extra2 [concrete = constants.%Extra2.type] {} {}
// CHECK:STDOUT:   %Extra3.decl: type = interface_decl @Extra3 [concrete = constants.%Extra3.type] {} {}
// CHECK:STDOUT:   %Extra4.decl: type = interface_decl @Extra4 [concrete = constants.%Extra4.type] {} {}
// CHECK:STDOUT:   %Extra5.decl: type = interface_decl @Extra5 [concrete = constants.%Extra5.type] {} {}
// CHECK:STDOUT:   %Extra6.decl: type = interface_decl @Extra6 [concrete = constants.%Extra6.type] {} {}
// CHECK:STDOUT:   %Extra7.decl: type = interface_decl @Extra7 [concrete = constants.%Extra7.type] {} {}
// CHECK:STDOUT:   %Extra8.decl: type = interface_decl @Extra8 [concrete = constants.%Extra8.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc13_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc13_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
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
// CHECK:STDOUT:     %.loc16_72: type = converted %.loc16_71, constants.%tuple.type.15d [concrete = constants.%tuple.type.15d]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.type.15d) [concrete = constants.%C.69b]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT:   %Self: %Extra1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.66c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT:   %Self: %Extra2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2ed]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT:   %Self: %Extra3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.622]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT:   %Self: %Extra4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.234]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT:   %Self: %Extra5.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d1b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT:   %Self: %Extra6.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.aeb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT:   %Self: %Extra7.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b7e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT:   %Self: %Extra8.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f90]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc14_26.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc14_26.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc14_26.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.2 [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc14_26.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc14_26.2: type = converted %Self.ref, %Self.as_type.loc14_26.2 [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc14_26.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.06f = fn_decl @F.2 [concrete = constants.%F.13c] {
// CHECK:STDOUT:     %self.patt: %C.69b = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.69b = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.69b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C [concrete = constants.%C.69b]
// CHECK:STDOUT:     %self: %C.69b = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc13_9.1: type) {
// CHECK:STDOUT:   %T.loc13_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc14_26.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_26.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc14_26.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C.69b]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc13_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_9.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc14_26.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type.15d) {
// CHECK:STDOUT:   %T.loc13_9.2 => constants.%tuple.type.15d
// CHECK:STDOUT:   %T.patt.loc13_9.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc14_26.1 => constants.%C.69b
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.42e: type = class_type @C, @C(type) [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.013: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%HasExtraInterfaces.import_ref.d54 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.013 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.074: type = class_type @C, @C(%tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %HasExtraInterfaces: <namespace> = namespace file.%HasExtraInterfaces.import, [concrete] {
// CHECK:STDOUT:     .C = %HasExtraInterfaces.C
// CHECK:STDOUT:     .I = %HasExtraInterfaces.I
// CHECK:STDOUT:     import HasExtraInterfaces//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces.C: %C.type = import_ref HasExtraInterfaces//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.5ab: type = import_ref HasExtraInterfaces//default, loc13_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.8f2: <witness> = import_ref HasExtraInterfaces//default, loc13_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.4c0 = import_ref HasExtraInterfaces//default, inst56 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.I: type = import_ref HasExtraInterfaces//default, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.e5d = import_ref HasExtraInterfaces//default, inst62 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.9cd: %I.assoc_type = import_ref HasExtraInterfaces//default, loc14_33, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %HasExtraInterfaces.F = import_ref HasExtraInterfaces//default, F, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.1db: %I.type = import_ref HasExtraInterfaces//default, inst62 [no loc], loaded [symbolic = constants.%Self.013]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.1c8 = import_ref HasExtraInterfaces//default, loc16_79, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.9c8 = import_ref HasExtraInterfaces//default, inst44 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.dfe = import_ref HasExtraInterfaces//default, inst40 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.6b6 = import_ref HasExtraInterfaces//default, inst36 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.576 = import_ref HasExtraInterfaces//default, inst32 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.0dd = import_ref HasExtraInterfaces//default, inst28 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.f83 = import_ref HasExtraInterfaces//default, inst24 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.975 = import_ref HasExtraInterfaces//default, inst20 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.a3c = import_ref HasExtraInterfaces//default, inst16 [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.aa8: type = import_ref HasExtraInterfaces//default, loc16_72, loaded [concrete = constants.%C.074]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.301: type = import_ref HasExtraInterfaces//default, loc16_77, loaded [concrete = constants.%I.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasExtraInterfaces = imports.%HasExtraInterfaces
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces.import = import HasExtraInterfaces
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %c.patt: %C.42e = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.42e = value_param_pattern %c.patt, call_param0
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
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%HasExtraInterfaces.import_ref.9cd
// CHECK:STDOUT:   witness = (imports.%HasExtraInterfaces.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.9c8
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.dfe
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.6b6
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.576
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.0dd
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.f83
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.975
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.a3c
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%HasExtraInterfaces.import_ref.aa8 as imports.%HasExtraInterfaces.import_ref.301 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%HasExtraInterfaces.import_ref.1c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%HasExtraInterfaces.import_ref.5ab: type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
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
// CHECK:STDOUT: fn @Test(%c.param_patt: %C.42e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.42e = name_ref c, %c
// CHECK:STDOUT:   %HasExtraInterfaces.ref.loc12: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [concrete = imports.%HasExtraInterfaces]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%HasExtraInterfaces.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%HasExtraInterfaces.import_ref.9cd [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(imports.%HasExtraInterfaces.import_ref.1db: %I.type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.013)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.013) {
// CHECK:STDOUT:   %Self => constants.%Self.013
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type) {
// CHECK:STDOUT:   %T => constants.%tuple.type
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
