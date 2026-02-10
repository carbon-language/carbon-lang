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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.92d: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.039: type = pattern_type %C [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {
// CHECK:STDOUT:     %self.patt: @HasF.F.%pattern_type (%pattern_type.92d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasF.F.%pattern_type (%pattern_type.92d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasF.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %C.ref as %HasF.ref {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.039 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.039 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.HasF.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = %HasF.impl_witness
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
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.92d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasF.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.92d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.039
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.26c: %HasG.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.ba6: type = symbolic_binding_type Self, 0, %Self.26c [symbolic]
// CHECK:STDOUT:   %pattern_type.348: type = pattern_type %Self.binding.as_type.ba6 [symbolic]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, @HasG.%HasG.G.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.e0d: <witness> = impl_witness @C.as.HasG.impl.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G.type: type = fn_type @C.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G: %C.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.435: %HasG.type = facet_value %C, (%HasG.impl_witness.e0d) [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.656: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @D.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e84: type = symbolic_binding_type Self, 0, %Self.656 [symbolic]
// CHECK:STDOUT:   %pattern_type.6ba: type = pattern_type %Self.binding.as_type.e84 [symbolic]
// CHECK:STDOUT:   %pattern_type.189: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F.type: type = fn_type @D.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %D.as.HasF.impl.F: %D.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %HasG.impl_witness.a94: <witness> = impl_witness @D.as.HasG.impl.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G.type: type = fn_type @D.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %D.as.HasG.impl.G: %D.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet.b67: %HasG.type = facet_value %D, (%HasG.impl_witness.a94) [concrete]
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
// CHECK:STDOUT:   %PackageA.import_ref.c44 = import_ref PackageA//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.182 = import_ref PackageA//default, loc4_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.0b3 = import_ref PackageA//default, loc5_21, unloaded
// CHECK:STDOUT:   %PackageA.F: %HasF.F.type = import_ref PackageA//default, F, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.d57: %HasF.type = import_ref PackageA//default, loc4_16, loaded [symbolic = constants.%Self.656]
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
// CHECK:STDOUT:   impl_decl @D.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.HasG.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic = constants.%Self.26c]
// CHECK:STDOUT:   %HasG.G.decl: %HasG.G.type = fn_decl @HasG.G [concrete = constants.%HasG.G] {
// CHECK:STDOUT:     %self.patt: @HasG.G.%pattern_type (%pattern_type.348) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @HasG.G.%pattern_type (%pattern_type.348) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @HasG.G.%Self.binding.as_type (%Self.binding.as_type.ba6) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ba6)] {
// CHECK:STDOUT:       %Self.ref: %HasG.type = name_ref Self, @HasG.%Self [symbolic = %Self (constants.%Self.26c)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ba6)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ba6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @HasG.G.%Self.binding.as_type (%Self.binding.as_type.ba6) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %HasG.G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%HasG.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.182
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.0b3
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: %C.ref as %HasG.ref {
// CHECK:STDOUT:   %C.as.HasG.impl.G.decl: %C.as.HasG.impl.G.type = fn_decl @C.as.HasG.impl.G [concrete = constants.%C.as.HasG.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8e5 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8e5 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.HasG.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%C.as.HasG.impl.G.decl), @C.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness %HasG.impl_witness_table [concrete = constants.%HasG.impl_witness.e0d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.HasG.impl.G.decl
// CHECK:STDOUT:   witness = %HasG.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasF.impl: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %D.as.HasF.impl.F.decl: %D.as.HasF.impl.F.type = fn_decl @D.as.HasF.impl.F [concrete = constants.%D.as.HasF.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.189 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.189 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @D.as.HasF.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%D.as.HasF.impl.F.decl), @D.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %D.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = %HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %D.as.HasG.impl.G.decl: %D.as.HasG.impl.G.type = fn_decl @D.as.HasG.impl.G [concrete = constants.%D.as.HasG.impl.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.189 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.189 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @D.as.HasG.impl.%D.ref [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%D.as.HasG.impl.G.decl), @D.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness %HasG.impl_witness_table [concrete = constants.%HasG.impl_witness.a94]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %D.as.HasG.impl.G.decl
// CHECK:STDOUT:   witness = %HasG.impl_witness
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
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.c44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.26c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ba6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.348)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @HasG.G.%Self.binding.as_type (%Self.binding.as_type.ba6));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.d57: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.656)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.6ba)]
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
// CHECK:STDOUT: specific @HasG.G(constants.%Self.26c) {
// CHECK:STDOUT:   %Self => constants.%Self.26c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.ba6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.348
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%HasG.facet.435) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.435
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self.656) {
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%HasG.facet.b67) {
// CHECK:STDOUT:   %Self => constants.%HasG.facet.b67
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [concrete]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.05f [concrete]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6ba: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness imports.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %.a26: type = fn_type_with_self_type %HasF.F.type, %HasF.facet [concrete]
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
// CHECK:STDOUT:   %PackageA.import_ref.c44 = import_ref PackageA//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.182 = import_ref PackageA//default, loc4_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.79c: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.05f: %HasF.F.type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.d57: %HasF.type = import_ref PackageA//default, loc4_16, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageA.import_ref.45d: <witness> = import_ref PackageA//default, loc11_16, loaded [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   %PackageA.import_ref.83d: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.4b0: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.a28: %C.as.HasF.impl.F.type = import_ref PackageA//default, loc12_22, loaded [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%PackageA.import_ref.a28), @C.as.HasF.impl [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.8e5 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.8e5 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref.loc6: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.182
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.79c
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: imports.%PackageA.import_ref.83d as imports.%PackageA.import_ref.4b0 [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.45d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.c44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc7: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.79c [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.a26 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.d57: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.6ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F [from "package_a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_df.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f09: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [concrete]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, imports.%PackageA.import_ref.05f [concrete]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6ba: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness imports.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %.95d: type = fn_type_with_self_type %HasF.F.type, %HasF.facet [concrete]
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
// CHECK:STDOUT:   %PackageB.import_ref.468 = import_ref PackageB//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageA.HasF: type = import_ref PackageA//default, HasF, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageA.import_ref.182 = import_ref PackageA//default, loc4_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.79c: %HasF.assoc_type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageA.F = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.05f: %HasF.F.type = import_ref PackageA//default, loc5_21, loaded [concrete = constants.%HasF.F]
// CHECK:STDOUT:   %PackageA.import_ref.d57: %HasF.type = import_ref PackageA//default, loc4_16, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageA.import_ref.907 = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %PackageA.import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageA.import_ref.c44 = import_ref PackageA//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageA.import_ref.83d: type = import_ref PackageA//default, loc11_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageA.import_ref.4b0: type = import_ref PackageA//default, loc11_11, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.3de: <witness> = import_ref PackageB//default, loc18_25, loaded [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.106: type = import_ref PackageB//default, loc18_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.7b1: type = import_ref PackageB//default, loc18_19, loaded [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %PackageB.import_ref.97f: %D.as.HasF.impl.F.type = import_ref PackageB//default, loc19_22, loaded [concrete = constants.%D.as.HasF.impl.F]
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%PackageB.import_ref.97f), @D.as.HasF.impl [concrete]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.f09 = value_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.f09 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%PackageB.D [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = value_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.182
// CHECK:STDOUT:   .F = imports.%PackageA.import_ref.79c
// CHECK:STDOUT:   witness = (imports.%PackageA.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: imports.%PackageA.import_ref.83d as imports.%PackageA.import_ref.4b0 [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageA.import_ref.907
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasF.impl: imports.%PackageB.import_ref.106 as imports.%PackageB.import_ref.7b1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.3de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.468
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.c44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDF(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%PackageA.HasF [concrete = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %HasF.assoc_type = name_ref F, imports.%PackageA.import_ref.79c [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.95d = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%D.as.HasF.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %D.as.HasF.impl.F.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(imports.%PackageA.import_ref.d57: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:   %Self: %HasF.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.6ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasF.impl.F [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.8e5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [concrete]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.d98 [concrete]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.588: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness imports.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %.549: type = fn_type_with_self_type %HasG.G.type, %HasG.facet [concrete]
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
// CHECK:STDOUT:   %PackageA.import_ref.c44 = import_ref PackageA//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.6dc = import_ref PackageB//default, loc6_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.239: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.d98: %HasG.G.type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%HasG.G]
// CHECK:STDOUT:   %PackageB.import_ref.945: %HasG.type = import_ref PackageB//default, loc6_16, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageB.import_ref.c85: <witness> = import_ref PackageB//default, loc13_25, loaded [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.2c1: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.d31a52.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.54b = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.468 = import_ref PackageB//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.106: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.d31a52.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.bb3: %C.as.HasG.impl.G.type = import_ref PackageB//default, loc14_22, loaded [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%PackageB.import_ref.bb3), @C.as.HasG.impl [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.8e5 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.8e5 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [concrete = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%PackageA.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.6dc
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.239
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: imports.%PackageB.import_ref.2c1 as imports.%PackageB.import_ref.d31a52.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.c85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: imports.%PackageB.import_ref.106 as imports.%PackageB.import_ref.d31a52.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageA.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageA.import_ref.c44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.468
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCG(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.239 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.549 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.HasG.impl.G.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(imports.%PackageB.import_ref.945: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.588)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.588
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_dg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f09: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [concrete]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, imports.%PackageB.import_ref.d98 [concrete]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.588: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness imports.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %D, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %.169: type = fn_type_with_self_type %HasG.G.type, %HasG.facet [concrete]
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
// CHECK:STDOUT:   %PackageB.import_ref.468 = import_ref PackageB//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageB.HasG: type = import_ref PackageB//default, HasG, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.6dc = import_ref PackageB//default, loc6_16, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.239: %HasG.assoc_type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageB.G = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.d98: %HasG.G.type = import_ref PackageB//default, loc7_21, loaded [concrete = constants.%HasG.G]
// CHECK:STDOUT:   %PackageB.import_ref.945: %HasG.type = import_ref PackageB//default, loc6_16, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageB.import_ref.148 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %PackageB.import_ref.8db: <witness> = import_ref PackageB//default, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageB.import_ref.572 = import_ref PackageB//default, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %PackageB.import_ref.2c1: type = import_ref PackageB//default, loc13_14, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %PackageB.import_ref.d31a52.1: type = import_ref PackageB//default, loc13_20, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.608: <witness> = import_ref PackageB//default, loc23_16, loaded [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:   %PackageB.import_ref.106: type = import_ref PackageB//default, loc23_6, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %PackageB.import_ref.d31a52.2: type = import_ref PackageB//default, loc23_11, loaded [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %PackageB.import_ref.83e: %D.as.HasG.impl.G.type = import_ref PackageB//default, loc24_22, loaded [concrete = constants.%D.as.HasG.impl.G]
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (%PackageB.import_ref.83e), @D.as.HasG.impl [concrete]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.f09 = value_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.f09 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %D.ref [concrete = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref.loc6: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%PackageB.D [concrete = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = value_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.6dc
// CHECK:STDOUT:   .G = imports.%PackageB.import_ref.239
// CHECK:STDOUT:   witness = (imports.%PackageB.G)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: imports.%PackageB.import_ref.2c1 as imports.%PackageB.import_ref.d31a52.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasG.impl: imports.%PackageB.import_ref.106 as imports.%PackageB.import_ref.d31a52.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageB.import_ref.608
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.468
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%PackageB.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageB.import_ref.572
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDG(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc7: <namespace> = name_ref PackageB, imports.%PackageB [concrete = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%PackageB.HasG [concrete = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %HasG.assoc_type = name_ref G, imports.%PackageB.import_ref.239 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.169 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%D.as.HasG.impl.G]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.ref, %impl.elem0
// CHECK:STDOUT:   %D.as.HasG.impl.G.call: init %empty_tuple.type = call %bound_method(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(imports.%PackageB.import_ref.945: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:   %Self: %HasG.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.588)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasG.impl.G [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.588
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.336: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z.H.type: type = fn_type @Z.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.H: %Z.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%Z.H.decl [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z.impl.%Z.impl_witness_table [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z.H.decl: %Z.H.type = fn_decl @Z.H [concrete = constants.%Z.H] {
// CHECK:STDOUT:     %self.patt: @Z.H.%pattern_type (%pattern_type.336) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Z.H.%pattern_type (%pattern_type.336) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Z.H.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Z.H.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.H.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %assoc0
// CHECK:STDOUT:   witness = (%Z.H.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.decl: %empty_tuple.type.as.Z.impl.H.type = fn_decl @empty_tuple.type.as.Z.impl.H [concrete = constants.%empty_tuple.type.as.Z.impl.H] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.Z.impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%empty_tuple.type.as.Z.impl.H.decl), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %empty_tuple.type.as.Z.impl.H.decl
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.H(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.336)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Z.H.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.Z.impl.H(%self.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, imports.%PackageAssociatedInterface.import_ref.9c5 [concrete]
// CHECK:STDOUT:   %Z.H.type: type = fn_type @Z.H [concrete]
// CHECK:STDOUT:   %Z.H: %Z.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.91a: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness imports.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %.d6a: type = fn_type_with_self_type %Z.H.type, %Z.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.type: type = fn_type @empty_tuple.type.as.Z.impl.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H: %empty_tuple.type.as.Z.impl.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.bound: <bound method> = bound_method %empty_tuple, %empty_tuple.type.as.Z.impl.H [concrete]
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
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.d8d = import_ref PackageAssociatedInterface//default, loc4_13, unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.35f: %Z.assoc_type = import_ref PackageAssociatedInterface//default, loc5_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %PackageAssociatedInterface.H = import_ref PackageAssociatedInterface//default, H, unloaded
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.9c5: %Z.H.type = import_ref PackageAssociatedInterface//default, loc5_21, loaded [concrete = constants.%Z.H]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.b0a: %Z.type = import_ref PackageAssociatedInterface//default, loc4_13, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.5f5: <witness> = import_ref PackageAssociatedInterface//default, loc8_14, loaded [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.e5c: type = import_ref PackageAssociatedInterface//default, loc8_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.5fa: type = import_ref PackageAssociatedInterface//default, loc8_12, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %PackageAssociatedInterface.import_ref.d43: %empty_tuple.type.as.Z.impl.H.type = import_ref PackageAssociatedInterface//default, loc9_22, loaded [concrete = constants.%empty_tuple.type.as.Z.impl.H]
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%PackageAssociatedInterface.import_ref.d43), @empty_tuple.type.as.Z.impl [concrete]
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
// CHECK:STDOUT:   .Self = imports.%PackageAssociatedInterface.import_ref.d8d
// CHECK:STDOUT:   .H = imports.%PackageAssociatedInterface.import_ref.35f
// CHECK:STDOUT:   witness = (imports.%PackageAssociatedInterface.H)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: imports.%PackageAssociatedInterface.import_ref.e5c as imports.%PackageAssociatedInterface.import_ref.5fa [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageAssociatedInterface.import_ref.5f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_4.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %PackageAssociatedInterface.ref: <namespace> = name_ref PackageAssociatedInterface, imports.%PackageAssociatedInterface [concrete = imports.%PackageAssociatedInterface]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, imports.%PackageAssociatedInterface.Z [concrete = constants.%Z.type]
// CHECK:STDOUT:   %H.ref: %Z.assoc_type = name_ref H, imports.%PackageAssociatedInterface.import_ref.35f [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.d6a = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.Z.impl.H]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc7_4.1, %impl.elem0 [concrete = constants.%empty_tuple.type.as.Z.impl.H.bound]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_4.2: %empty_tuple.type = converted %.loc7_4.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.type.as.Z.impl.H.call: init %empty_tuple.type = call %bound_method(%.loc7_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.H(imports.%PackageAssociatedInterface.import_ref.b0a: %Z.type) [from "associated_interface.carbon"] {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.91a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.Z.impl.H [from "associated_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.H(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.91a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.880: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y.K.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic]
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
// CHECK:STDOUT:     %X.patt: @AnyParam.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern X, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4: type = splice_block %T.ref [symbolic = %T.loc4_16.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_16.2 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc4_26.2: @AnyParam.%T.loc4_16.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc4_26.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Y.K.decl: %Y.K.type = fn_decl @Y.K [concrete = constants.%Y.K] {
// CHECK:STDOUT:     %self.patt: @Y.K.%pattern_type (%pattern_type.880) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Y.K.%pattern_type (%pattern_type.880) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Y.K.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_14.1: type = splice_block %.loc7_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Y.type = name_ref Self, @Y.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Y.K.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, %Y.K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%Y.K.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(%T.loc4_16.2: type, %X.loc4_26.2: @AnyParam.%T.loc4_16.1 (%T)) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   %X.loc4_26.1: @AnyParam.%T.loc4_16.1 (%T) = symbolic_binding X, 1 [symbolic = %X.loc4_26.1 (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_16.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
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
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.880)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Y.K.%Self.binding.as_type (%Self.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:   %X.loc4_26.1 => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.880
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericInterface.type.9e8: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.9e8 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericInterface.type.ad9: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.383: %GenericInterface.type.ad9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.80a: type = pattern_type %GenericInterface.type.9e8 [concrete]
// CHECK:STDOUT:   %AnyParam.33d: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.9e8, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.079: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @AnyParam.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.c23: type = symbolic_binding_type Self, 0, %Self.079 [symbolic]
// CHECK:STDOUT:   %pattern_type.420: type = pattern_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %pattern_type.a39: type = pattern_type %AnyParam.33d [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.33d, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.33d = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.749: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.e48 [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.535 = import_ref PackageHasParam//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.b3b: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.b42: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.9b4 = import_ref PackageHasParam//default, loc6_13, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.7ef: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %PackageHasParam.K: %Y.K.type = import_ref PackageHasParam//default, K, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.71d: %Y.type = import_ref PackageHasParam//default, loc6_13, loaded [symbolic = constants.%Self.079]
// CHECK:STDOUT:   %PackageHasParam.import_ref.e48: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
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
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.9e8 = interface_decl @GenericInterface [concrete = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc6_28.2: type = symbolic_binding U, 0 [symbolic = %U.loc6_28.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @AnyParam.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.9e8 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.9e8, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.33d]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_52: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%U.loc6_28.2: type) {
// CHECK:STDOUT:   %U.loc6_28.1: type = symbolic_binding U, 0 [symbolic = %U.loc6_28.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U.loc6_28.1)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.ad9)]
// CHECK:STDOUT:   %Self.loc6_38.2: @GenericInterface.%GenericInterface.type (%GenericInterface.type.ad9) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.383)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_38.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.ad9) = symbolic_binding Self, 1 [symbolic = %Self.loc6_38.2 (constants.%Self.383)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc6_38.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.9b4
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.7ef
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.decl: %AnyParam.as.Y.impl.K.type = fn_decl @AnyParam.as.Y.impl.K [concrete = constants.%AnyParam.as.Y.impl.K] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a39 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a39 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.33d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @AnyParam.as.Y.impl.%AnyParam [concrete = constants.%AnyParam.33d]
// CHECK:STDOUT:     %self: %AnyParam.33d = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%AnyParam.as.Y.impl.K.decl), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %AnyParam.as.Y.impl.K.decl
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.b3b: type, imports.%PackageHasParam.import_ref.b42: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = symbolic_binding X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.535
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.71d: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.079)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c23)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.420)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K(%self.param: %AnyParam.33d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.a39 = ref_binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.a39 = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.33d = var %obj.var_patt
// CHECK:STDOUT:   %.loc13_58.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_58.2: init %AnyParam.33d to %obj.var = class_init () [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3: init %AnyParam.33d = converted %.loc13_58.1, %.loc13_58.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3
// CHECK:STDOUT:   %.loc13_53: type = splice_block %AnyParam [concrete = constants.%AnyParam.33d] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.9e8 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.9e8, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.33d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.33d = ref_binding obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.33d = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.7ef [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %impl.elem0: %.94f = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.33d = acquire_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method(%.loc14)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %obj.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%obj.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %AnyParam.33d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.9e8, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.9e8
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.079) {
// CHECK:STDOUT:   %Self => constants.%Self.079
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.c23
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.420
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%AnyParam.33d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a39
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.0da: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.b26: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.793: %GenericInterface.type.b26 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.a647: type = pattern_type %GenericInterface.type.0da [concrete]
// CHECK:STDOUT:   %AnyParam.861: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.0da, %GenericInterface.generic) [concrete]
// CHECK:STDOUT:   %pattern_type.3ad: type = pattern_type %AnyParam.861 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.861 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.079: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.749: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.e48 [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.c23: type = symbolic_binding_type Self, 0, %Self.079 [symbolic]
// CHECK:STDOUT:   %pattern_type.420: type = pattern_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness imports.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.861, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %.854: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.535 = import_ref PackageHasParam//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.b3b: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.b42: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageGenericInterface.GenericInterface: %GenericInterface.type.0da = import_ref PackageGenericInterface//default, GenericInterface, loaded [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.116 = import_ref PackageGenericInterface//default, loc6_38, unloaded
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.b3b: type = import_ref PackageGenericInterface//default, loc6_28, loaded [symbolic = @GenericInterface.%U (constants.%U)]
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.9b4 = import_ref PackageHasParam//default, loc6_13, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.7ef: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.e48: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.71d: %Y.type = import_ref PackageHasParam//default, loc6_13, loaded [symbolic = constants.%Self.079]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.564: <witness> = import_ref PackageGenericInterface//default, loc8_70, loaded [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.2fb: type = import_ref PackageGenericInterface//default, loc8_47, loaded [concrete = constants.%AnyParam.861]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.6c1: type = import_ref PackageGenericInterface//default, loc8_67, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageGenericInterface.import_ref.ac9: %AnyParam.as.Y.impl.K.type = import_ref PackageGenericInterface//default, loc9_22, loaded [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%PackageGenericInterface.import_ref.ac9), @AnyParam.as.Y.impl [concrete]
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
// CHECK:STDOUT: generic interface @GenericInterface(imports.%PackageGenericInterface.import_ref.b3b: type) [from "has_generic_interface.carbon"] {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.b26)]
// CHECK:STDOUT:   %Self: @GenericInterface.%GenericInterface.type (%GenericInterface.type.b26) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.793)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageGenericInterface.import_ref.116
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.9b4
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.7ef
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: imports.%PackageGenericInterface.import_ref.2fb as imports.%PackageGenericInterface.import_ref.6c1 [from "has_generic_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericInterface.import_ref.564
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.b3b: type, imports.%PackageHasParam.import_ref.b42: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = symbolic_binding X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.535
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.3ad = ref_binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.3ad = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.861 = var %obj.var_patt
// CHECK:STDOUT:   %.loc9_50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_50.2: init %AnyParam.861 to %obj.var = class_init () [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8: init %AnyParam.861 = converted %.loc9_50.1, %.loc9_50.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8
// CHECK:STDOUT:   %.loc9_45: type = splice_block %AnyParam [concrete = constants.%AnyParam.861] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %PackageGenericInterface.ref: <namespace> = name_ref PackageGenericInterface, imports.%PackageGenericInterface [concrete = imports.%PackageGenericInterface]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.0da = name_ref GenericInterface, imports.%PackageGenericInterface.GenericInterface [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) [concrete = constants.%AnyParam.861]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.861 = ref_binding obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.861 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc10: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.7ef [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %impl.elem0: %.854 = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc10: %AnyParam.861 = acquire_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method(%.loc10)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %obj.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%obj.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.71d: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.079)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c23)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.420)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K [from "has_generic_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %AnyParam.861) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.0da
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a647
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.079) {
// CHECK:STDOUT:   %Self => constants.%Self.079
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.c23
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.420
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass: type = class_type @GenericClass, @GenericClass(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [concrete]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.483: type = pattern_type %GenericClass.type [concrete]
// CHECK:STDOUT:   %AnyParam.8e0: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.079: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @AnyParam.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.c23: type = symbolic_binding_type Self, 0, %Self.079 [symbolic]
// CHECK:STDOUT:   %pattern_type.420: type = pattern_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff6: type = pattern_type %AnyParam.8e0 [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.8e0, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %L.type: type = fn_type @L [concrete]
// CHECK:STDOUT:   %L: %L.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.8e0 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.749: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.e48 [concrete]
// CHECK:STDOUT:   %.dfd: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.535 = import_ref PackageHasParam//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.b3b: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.b42: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.9b4 = import_ref PackageHasParam//default, loc6_13, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.7ef: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %PackageHasParam.K: %Y.K.type = import_ref PackageHasParam//default, K, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.71d: %Y.type = import_ref PackageHasParam//default, loc6_13, loaded [symbolic = constants.%Self.079]
// CHECK:STDOUT:   %PackageHasParam.import_ref.e48: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc6_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc6_20.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @AnyParam.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.8e0]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_48: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.9b4
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.7ef
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.decl: %AnyParam.as.Y.impl.K.type = fn_decl @AnyParam.as.Y.impl.K [concrete = constants.%AnyParam.as.Y.impl.K] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ff6 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ff6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AnyParam.8e0 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @AnyParam.as.Y.impl.%AnyParam [concrete = constants.%AnyParam.8e0]
// CHECK:STDOUT:     %self: %AnyParam.8e0 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%AnyParam.as.Y.impl.K.decl), @AnyParam.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %AnyParam.as.Y.impl.K.decl
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericClass(%U.loc6_20.2: type) {
// CHECK:STDOUT:   %U.loc6_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc6_20.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericClass
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.b3b: type, imports.%PackageHasParam.import_ref.b42: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = symbolic_binding X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.535
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.71d: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.079)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c23)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.420)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K(%self.param: %AnyParam.8e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.ff6 = ref_binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.ff6 = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.8e0 = var %obj.var_patt
// CHECK:STDOUT:   %.loc13_54.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_54.2: init %AnyParam.8e0 to %obj.var = class_init () [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_3: init %AnyParam.8e0 = converted %.loc13_54.1, %.loc13_54.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_3
// CHECK:STDOUT:   %.loc13_49: type = splice_block %AnyParam [concrete = constants.%AnyParam.8e0] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.8e0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.8e0 = ref_binding obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.8e0 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.7ef [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %impl.elem0: %.dfd = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14: %AnyParam.8e0 = acquire_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method(%.loc14)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %obj.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%obj.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %AnyParam.8e0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericClass(constants.%U) {
// CHECK:STDOUT:   %U.loc6_20.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) {
// CHECK:STDOUT:   %T => constants.%GenericClass.type
// CHECK:STDOUT:   %X => constants.%GenericClass.generic
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.483
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Self.079) {
// CHECK:STDOUT:   %Self => constants.%Self.079
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.c23
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.420
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.K(constants.%Y.facet) {
// CHECK:STDOUT:   %Self => constants.%Y.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%AnyParam.8e0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff6
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = generic_class_type @GenericClass [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %GenericClass.type [concrete]
// CHECK:STDOUT:   %AnyParam.d71: type = class_type @AnyParam, @AnyParam(%GenericClass.type, %GenericClass.generic) [concrete]
// CHECK:STDOUT:   %pattern_type.312: type = pattern_type %AnyParam.d71 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.d71 = struct_value () [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.079: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.749: %Y.assoc_type = assoc_entity element0, imports.%PackageHasParam.import_ref.e48 [concrete]
// CHECK:STDOUT:   %Y.K.type: type = fn_type @Y.K [concrete]
// CHECK:STDOUT:   %Y.K: %Y.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.c23: type = symbolic_binding_type Self, 0, %Self.079 [symbolic]
// CHECK:STDOUT:   %pattern_type.420: type = pattern_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type.c23 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness imports.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.d71, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %.536: type = fn_type_with_self_type %Y.K.type, %Y.facet [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.type: type = fn_type @AnyParam.as.Y.impl.K [concrete]
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K: %AnyParam.as.Y.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %PackageHasParam.import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.535 = import_ref PackageHasParam//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.b3b: type = import_ref PackageHasParam//default, loc4_16, loaded [symbolic = @AnyParam.%T (constants.%T)]
// CHECK:STDOUT:   %PackageHasParam.import_ref.b42: @AnyParam.%T (%T) = import_ref PackageHasParam//default, loc4_26, loaded [symbolic = @AnyParam.%X (constants.%X)]
// CHECK:STDOUT:   %PackageGenericClass.GenericClass: %GenericClass.type = import_ref PackageGenericClass//default, GenericClass, loaded [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.8f2: <witness> = import_ref PackageGenericClass//default, loc6_31, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.7ce = import_ref PackageGenericClass//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %PackageGenericClass.import_ref.b3b: type = import_ref PackageGenericClass//default, loc6_20, loaded [symbolic = @GenericClass.%U (constants.%U)]
// CHECK:STDOUT:   %PackageHasParam.Y: type = import_ref PackageHasParam//default, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageHasParam.import_ref.9b4 = import_ref PackageHasParam//default, loc6_13, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.7ef: %Y.assoc_type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %PackageHasParam.K = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %PackageHasParam.import_ref.e48: %Y.K.type = import_ref PackageHasParam//default, loc7_22, loaded [concrete = constants.%Y.K]
// CHECK:STDOUT:   %PackageHasParam.import_ref.71d: %Y.type = import_ref PackageHasParam//default, loc6_13, loaded [symbolic = constants.%Self.079]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.f5f: <witness> = import_ref PackageGenericClass//default, loc8_66, loaded [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.1b3: type = import_ref PackageGenericClass//default, loc8_43, loaded [concrete = constants.%AnyParam.d71]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.6c1: type = import_ref PackageGenericClass//default, loc8_63, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %PackageGenericClass.import_ref.6b4: %AnyParam.as.Y.impl.K.type = import_ref PackageGenericClass//default, loc9_22, loaded [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%PackageGenericClass.import_ref.6b4), @AnyParam.as.Y.impl [concrete]
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
// CHECK:STDOUT:   .Self = imports.%PackageHasParam.import_ref.9b4
// CHECK:STDOUT:   .K = imports.%PackageHasParam.import_ref.7ef
// CHECK:STDOUT:   witness = (imports.%PackageHasParam.K)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AnyParam.as.Y.impl: imports.%PackageGenericClass.import_ref.1b3 as imports.%PackageGenericClass.import_ref.6c1 [from "has_generic_class.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%PackageGenericClass.import_ref.f5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(imports.%PackageHasParam.import_ref.b3b: type, imports.%PackageHasParam.import_ref.b42: @AnyParam.%T (%T)) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %X: @AnyParam.%T (%T) = symbolic_binding X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageHasParam.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageHasParam.import_ref.535
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericClass(imports.%PackageGenericClass.import_ref.b3b: type) [from "has_generic_class.carbon"] {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%PackageGenericClass.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%PackageGenericClass.import_ref.7ce
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %obj.patt: %pattern_type.312 = ref_binding_pattern obj [concrete]
// CHECK:STDOUT:     %obj.var_patt: %pattern_type.312 = var_pattern %obj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj.var: ref %AnyParam.d71 = var %obj.var_patt
// CHECK:STDOUT:   %.loc8_74.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_74.2: init %AnyParam.d71 to %obj.var = class_init () [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc8_3: init %AnyParam.d71 = converted %.loc8_74.1, %.loc8_74.2 [concrete = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_69: type = splice_block %AnyParam [concrete = constants.%AnyParam.d71] {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%PackageHasParam.AnyParam [concrete = constants.%AnyParam.generic]
// CHECK:STDOUT:     %PackageGenericClass.ref: <namespace> = name_ref PackageGenericClass, imports.%PackageGenericClass [concrete = imports.%PackageGenericClass]
// CHECK:STDOUT:     %GenericClass.ref: %GenericClass.type = name_ref GenericClass, imports.%PackageGenericClass.GenericClass [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericClass.type, constants.%GenericClass.generic) [concrete = constants.%AnyParam.d71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %obj: ref %AnyParam.d71 = ref_binding obj, %obj.var
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.d71 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc9: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [concrete = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%PackageHasParam.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %Y.assoc_type = name_ref K, imports.%PackageHasParam.import_ref.7ef [concrete = constants.%assoc0.749]
// CHECK:STDOUT:   %impl.elem0: %.536 = impl_witness_access constants.%Y.impl_witness, element0 [concrete = constants.%AnyParam.as.Y.impl.K]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %obj.ref, %impl.elem0
// CHECK:STDOUT:   %.loc9: %AnyParam.d71 = acquire_value %obj.ref
// CHECK:STDOUT:   %AnyParam.as.Y.impl.K.call: init %empty_tuple.type = call %bound_method(%.loc9)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %obj.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%obj.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.K(imports.%PackageHasParam.import_ref.71d: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.079)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c23)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.420)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AnyParam.as.Y.impl.K [from "has_generic_class.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %AnyParam.d71) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
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
// CHECK:STDOUT: specific @Y.K(constants.%Self.079) {
// CHECK:STDOUT:   %Self => constants.%Self.079
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.c23
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.420
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %Self.244: %Extra1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Self.b9c: %Extra2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Self.409: %Extra3.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Self.035: %Extra4.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Self.28e: %Extra5.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Self.b11: %Extra6.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Self.fa4: %Extra7.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Self.8a2: %Extra8.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.ebb: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.c1b: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c1b [symbolic]
// CHECK:STDOUT:   %pattern_type.c3e: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %tuple.type.c53: type = tuple_type (type, type, type, type, type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c53 = tuple_value (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %tuple.type.9c0: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.806: type = class_type @C, @C(%tuple.type.9c0) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e9e: type = pattern_type %C.806 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.806, (%I.impl_witness) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc13_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_9.1 (constants.%T)]
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
// CHECK:STDOUT:     %.loc16_71: %tuple.type.c53 = tuple_literal (%Extra1.ref, %Extra2.ref, %Extra3.ref, %Extra4.ref, %Extra5.ref, %Extra6.ref, %Extra7.ref, %Extra8.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc16_72: type = converted %.loc16_71, constants.%tuple.type.9c0 [concrete = constants.%tuple.type.9c0]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.type.9c0) [concrete = constants.%C.806]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT:   %Self: %Extra1.type = symbolic_binding Self, 0 [symbolic = constants.%Self.244]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT:   %Self: %Extra2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b9c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT:   %Self: %Extra3.type = symbolic_binding Self, 0 [symbolic = constants.%Self.409]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT:   %Self: %Extra4.type = symbolic_binding Self, 0 [symbolic = constants.%Self.035]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT:   %Self: %Extra5.type = symbolic_binding Self, 0 [symbolic = constants.%Self.28e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT:   %Self: %Extra6.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b11]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT:   %Self: %Extra7.type = symbolic_binding Self, 0 [symbolic = constants.%Self.fa4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT:   %Self: %Extra8.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c1b]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %self.patt: @I.F.%pattern_type (%pattern_type.c3e) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.F.%pattern_type (%pattern_type.c3e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc14_26.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e9e = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e9e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.806 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C [concrete = constants.%C.806]
// CHECK:STDOUT:     %self: %C.806 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc13_9.2: type) {
// CHECK:STDOUT:   %T.loc13_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.ebb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.c3e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C.806) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc13_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.c1b) {
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type.9c0) {
// CHECK:STDOUT:   %T.loc13_9.1 => constants.%tuple.type.9c0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.806
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e9e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.42e: type = class_type @C, @C(type) [concrete]
// CHECK:STDOUT:   %pattern_type.6e4: type = pattern_type %C.42e [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.d36: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%HasExtraInterfaces.import_ref.795 [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.d36 [symbolic]
// CHECK:STDOUT:   %pattern_type.327: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [concrete]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [concrete]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [concrete]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [concrete]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [concrete]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [concrete]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [concrete]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [concrete]
// CHECK:STDOUT:   %C.317: type = class_type @C, @C(%tuple.type) [concrete]
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
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.8f2: <witness> = import_ref HasExtraInterfaces//default, loc13_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.87a = import_ref HasExtraInterfaces//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.b3b: type = import_ref HasExtraInterfaces//default, loc13_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %HasExtraInterfaces.I: type = import_ref HasExtraInterfaces//default, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.027 = import_ref HasExtraInterfaces//default, loc14_13, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.f9b: %I.assoc_type = import_ref HasExtraInterfaces//default, loc14_33, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %HasExtraInterfaces.F = import_ref HasExtraInterfaces//default, F, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.795: %I.F.type = import_ref HasExtraInterfaces//default, loc14_33, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.1d0: %I.type = import_ref HasExtraInterfaces//default, loc14_13, loaded [symbolic = constants.%Self.d36]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.afb = import_ref HasExtraInterfaces//default, loc16_79, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.040 = import_ref HasExtraInterfaces//default, loc11_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.623 = import_ref HasExtraInterfaces//default, loc10_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.32b = import_ref HasExtraInterfaces//default, loc9_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.1f2 = import_ref HasExtraInterfaces//default, loc8_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.ba2 = import_ref HasExtraInterfaces//default, loc7_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.477 = import_ref HasExtraInterfaces//default, loc6_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.b19 = import_ref HasExtraInterfaces//default, loc5_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.78f = import_ref HasExtraInterfaces//default, loc4_18, unloaded
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.635: type = import_ref HasExtraInterfaces//default, loc16_72, loaded [concrete = constants.%C.317]
// CHECK:STDOUT:   %HasExtraInterfaces.import_ref.39b: type = import_ref HasExtraInterfaces//default, loc16_77, loaded [concrete = constants.%I.type]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.6e4 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6e4 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.42e = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %C [concrete = constants.%C.42e] {
// CHECK:STDOUT:       %HasExtraInterfaces.ref.loc5: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [concrete = imports.%HasExtraInterfaces]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%HasExtraInterfaces.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type) [concrete = constants.%C.42e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.42e = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.027
// CHECK:STDOUT:   .F = imports.%HasExtraInterfaces.import_ref.f9b
// CHECK:STDOUT:   witness = (imports.%HasExtraInterfaces.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.040
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.623
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.32b
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.1f2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.ba2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.477
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.b19
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%HasExtraInterfaces.import_ref.78f
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%HasExtraInterfaces.import_ref.635 as imports.%HasExtraInterfaces.import_ref.39b [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%HasExtraInterfaces.import_ref.afb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%HasExtraInterfaces.import_ref.b3b: type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%HasExtraInterfaces.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%HasExtraInterfaces.import_ref.87a
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%c.param: %C.42e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.42e = name_ref c, %c
// CHECK:STDOUT:   %HasExtraInterfaces.ref.loc12: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [concrete = imports.%HasExtraInterfaces]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%HasExtraInterfaces.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%HasExtraInterfaces.import_ref.f9b [concrete = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%HasExtraInterfaces.import_ref.1d0: %I.type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d36)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.327)]
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
// CHECK:STDOUT: specific @I.F(constants.%Self.d36) {
// CHECK:STDOUT:   %Self => constants.%Self.d36
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.327
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type) {
// CHECK:STDOUT:   %T => constants.%tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
