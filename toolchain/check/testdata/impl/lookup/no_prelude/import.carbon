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
  fn F();
}

class C {}

// Same library as the class and interface.
impl C as HasF {
  fn F() {}
}

// --- package_b.carbon

package PackageB;

import PackageA;

interface HasG {
  fn G();
}

class D {}

// Same library as the interface.
impl PackageA.C as HasG {
  fn G() {}
}

// Same library as the class.
impl D as PackageA.HasF {
  fn F() {}
}

// Same library as the class and interface.
impl D as HasG {
  fn G() {}
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
  fn H();
}

impl () as Z {
  fn H() {}
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
  fn K() {}
}

// --- has_generic_interface.carbon

package PackageGenericInterface;

import PackageHasParam;

interface GenericInterface(U:! type) {}

impl PackageHasParam.AnyParam(GenericInterface) as PackageHasParam.Y {
  fn K() {}
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
interface I { fn F(); }

impl C((Extra1, Extra2, Extra3, Extra4, Extra5, Extra6, Extra7, Extra8)) as I {
  fn F() {}
}

// --- fail_use_has_extra_interfaces.carbon

package UseHasExtraInterfaces;
import HasExtraInterfaces;

fn Test(c: HasExtraInterfaces.C(type)) {
  // This triggers the import of a bunch more interfaces, which reallocates the
  // interface ValueStore. Ensure that doesn't result in a use-after-free crash.
  // CHECK:STDERR: fail_use_has_extra_interfaces.carbon:[[@LINE+3]]:3: error: cannot access member of interface `HasExtraInterfaces.I` in type `HasExtraInterfaces.C(type)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c.(HasExtraInterfaces.I.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  c.(HasExtraInterfaces.I.F)();
}

// CHECK:STDOUT: --- package_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.16d: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.a20: %F.type.16d = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16d [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.2e0: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.89c: %F.type.2e0 = struct_value () [template]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.16d = fn_decl @F.1 [template = constants.%F.a20] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2e0 = fn_decl @F.2 [template = constants.%F.89c] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.4db: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.6fe: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.0ea: %G.type.6fe = struct_value () [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %HasG.type, %G.type.6fe [template]
// CHECK:STDOUT:   %assoc0: %G.assoc_type = assoc_entity element0, @HasG.%G.decl [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %impl_witness.58b: <witness> = impl_witness (@impl.1.%G.decl) [template]
// CHECK:STDOUT:   %G.type.407: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.b4a: %G.type.407 = struct_value () [template]
// CHECK:STDOUT:   %HasG.facet.fd4: %HasG.type = facet_value %C, %impl_witness.58b [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.c46: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.d75: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.3ac: %F.type.d75 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.ea3: <witness> = impl_witness (@impl.2.%F.decl) [template]
// CHECK:STDOUT:   %F.type.e3c: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.66f: %F.type.e3c = struct_value () [template]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %D, %impl_witness.ea3 [template]
// CHECK:STDOUT:   %impl_witness.88f: <witness> = impl_witness (@impl.3.%G.decl) [template]
// CHECK:STDOUT:   %G.type.854: type = fn_type @G.3 [template]
// CHECK:STDOUT:   %G.ae6: %G.type.854 = struct_value () [template]
// CHECK:STDOUT:   %HasG.facet.bec: %HasG.type = facet_value %D, %impl_witness.88f [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.0ac
// CHECK:STDOUT:     .HasF = %import_ref.2d0
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.0ac: type = import_ref PackageA//default, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref PackageA//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2d0: type = import_ref PackageA//default, HasF, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.28d = import_ref PackageA//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.cc4 = import_ref PackageA//default, loc5_9, unloaded
// CHECK:STDOUT:   %import_ref.ab3: %F.type.d75 = import_ref PackageA//default, F, loaded [template = constants.%F.3ac]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [template = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.1.%G.decl) [template = constants.%impl_witness.58b]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, imports.%import_ref.2d0 [template = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.2.%F.decl) [template = constants.%impl_witness.ea3]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.3.%G.decl) [template = constants.%impl_witness.88f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4db]
// CHECK:STDOUT:   %G.decl: %G.type.6fe = fn_decl @G.1 [template = constants.%G.0ea] {} {}
// CHECK:STDOUT:   %assoc0: %G.assoc_type = assoc_entity element0, %G.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.28d
// CHECK:STDOUT:   .F = imports.%import_ref.cc4
// CHECK:STDOUT:   witness = (imports.%import_ref.ab3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.407 = fn_decl @G.2 [template = constants.%G.b4a] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.e3c = fn_decl @F.2 [template = constants.%F.66f] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.854 = fn_decl @G.3 [template = constants.%G.ae6] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self.c46: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.4db) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet.fd4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.c46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet.bec) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.d75: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.d75 [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, imports.%import_ref.4a0 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.ac3) [template]
// CHECK:STDOUT:   %F.type.c5e: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.8d0: %F.type.c5e = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.0ac
// CHECK:STDOUT:     .HasF = %import_ref.2d0
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.0ac: type = import_ref PackageA//default, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref PackageA//default, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref PackageA//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2d0: type = import_ref PackageA//default, HasF, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.28d = import_ref PackageA//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b67: %F.assoc_type = import_ref PackageA//default, loc5_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.64f = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %import_ref.7a7: <witness> = import_ref PackageA//default, loc11_16, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.3de: type = import_ref PackageA//default, loc11_6, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e33: type = import_ref PackageA//default, loc11_11, loaded [template = constants.%HasF.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .TestCF = %TestCF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %TestCF.decl: %TestCF.type = fn_decl @TestCF [template = constants.%TestCF] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [template = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref.loc6: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.28d
// CHECK:STDOUT:   .F = imports.%import_ref.b67
// CHECK:STDOUT:   witness = (imports.%import_ref.64f)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.3de as imports.%import_ref.e33 [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.7a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc7: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.2d0 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, imports.%import_ref.b67 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %F.type.d75 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.8d0]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "package_a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_df.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.c46: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.d75: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.d75 [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, imports.%import_ref.4a0 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.e33a) [template]
// CHECK:STDOUT:   %F.type.975: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.ec0: %F.type.975 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .HasF = %import_ref.2d0
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.275
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.275: type = import_ref PackageB//default, D, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref PackageB//default, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.cab = import_ref PackageB//default, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2d0: type = import_ref PackageA//default, HasF, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.28d = import_ref PackageA//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b67: %F.assoc_type = import_ref PackageA//default, loc5_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.64f = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %import_ref.487 = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref PackageA//default, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref PackageA//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3de: type = import_ref PackageA//default, loc11_6, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e338: type = import_ref PackageA//default, loc11_11, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.f91 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %import_ref.f3e = import_ref PackageB//default, inst17 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.a79 = import_ref PackageB//default, loc7_9, unloaded
// CHECK:STDOUT:   %import_ref.f03 = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %import_ref.4a6: type = import_ref PackageB//default, loc13_14, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e527c5.1: type = import_ref PackageB//default, loc13_20, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.dc8: <witness> = import_ref PackageB//default, loc18_25, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.3b992b.1: type = import_ref PackageB//default, loc18_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.9e1: type = import_ref PackageB//default, loc18_19, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.ebd = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %import_ref.3b992b.2: type = import_ref PackageB//default, loc23_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.e527c5.2: type = import_ref PackageB//default, loc23_11, loaded [template = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDF = %TestDF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDF.decl: %TestDF.type = fn_decl @TestDF [template = constants.%TestDF] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %D.ref [template = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.28d
// CHECK:STDOUT:   .F = imports.%import_ref.b67
// CHECK:STDOUT:   witness = (imports.%import_ref.64f)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f3e
// CHECK:STDOUT:   .G = imports.%import_ref.a79
// CHECK:STDOUT:   witness = (imports.%import_ref.f03)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.3de as imports.%import_ref.e338 [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.487
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.4a6 as imports.%import_ref.e527c5.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.3b992b.1 as imports.%import_ref.9e1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.dc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.3b992b.2 as imports.%import_ref.e527c5.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.ebd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.cab
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDF(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.2d0 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, imports.%import_ref.b67 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %F.type.d75 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.ec0]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self.c46: %HasF.type) [from "package_a.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.c46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.102: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.813: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %HasG.type, %G.type.813 [template]
// CHECK:STDOUT:   %assoc0: %G.assoc_type = assoc_entity element0, imports.%import_ref.5b2 [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.a5c) [template]
// CHECK:STDOUT:   %G.type.6c1: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.b51: %G.type.6c1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.0ac
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .HasG = %import_ref.9db
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.0ac: type = import_ref PackageA//default, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref PackageA//default, loc8_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref PackageA//default, inst25 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9db: type = import_ref PackageB//default, HasG, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.f3e = import_ref PackageB//default, inst17 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.89f: %G.assoc_type = import_ref PackageB//default, loc7_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.f03 = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %import_ref.487 = import_ref PackageA//default, loc11_16, unloaded
// CHECK:STDOUT:   %import_ref.28d = import_ref PackageA//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.cc4 = import_ref PackageA//default, loc5_9, unloaded
// CHECK:STDOUT:   %import_ref.64f = import_ref PackageA//default, F, unloaded
// CHECK:STDOUT:   %import_ref.3de: type = import_ref PackageA//default, loc11_6, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e33: type = import_ref PackageA//default, loc11_11, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.38a: <witness> = import_ref PackageB//default, loc13_25, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.4a6: type = import_ref PackageB//default, loc13_14, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e527c5.1: type = import_ref PackageB//default, loc13_20, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.f77 = import_ref PackageB//default, loc18_25, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref PackageB//default, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.cab = import_ref PackageB//default, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3b992b.1: type = import_ref PackageB//default, loc18_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.9e1: type = import_ref PackageB//default, loc18_19, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.ebd = import_ref PackageB//default, loc23_16, unloaded
// CHECK:STDOUT:   %import_ref.3b992b.2: type = import_ref PackageB//default, loc23_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.e527c5.2: type = import_ref PackageB//default, loc23_11, loaded [template = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestCG = %TestCG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestCG.decl: %TestCG.type = fn_decl @TestCG [template = constants.%TestCG] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [template = constants.%C] {
// CHECK:STDOUT:       %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f3e
// CHECK:STDOUT:   .G = imports.%import_ref.89f
// CHECK:STDOUT:   witness = (imports.%import_ref.f03)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.28d
// CHECK:STDOUT:   .F = imports.%import_ref.cc4
// CHECK:STDOUT:   witness = (imports.%import_ref.64f)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.3de as imports.%import_ref.e33 [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.487
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.4a6 as imports.%import_ref.e527c5.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.38a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.3b992b.1 as imports.%import_ref.9e1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.3b992b.2 as imports.%import_ref.e527c5.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.ebd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.cab
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCG(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.9db [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %G.assoc_type = name_ref G, imports.%import_ref.89f [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %G.type.813 = impl_witness_access constants.%impl_witness, element0 [template = constants.%G.b51]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(constants.%Self.102: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.102) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_dg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.102: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.813: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %HasG.type, %G.type.813 [template]
// CHECK:STDOUT:   %assoc0: %G.assoc_type = assoc_entity element0, imports.%import_ref.5b2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.d18) [template]
// CHECK:STDOUT:   %G.type.ec8: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.e97: %G.type.ec8 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.275
// CHECK:STDOUT:     .HasG = %import_ref.9db
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.275: type = import_ref PackageB//default, D, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref PackageB//default, loc10_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.cab = import_ref PackageB//default, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9db: type = import_ref PackageB//default, HasG, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.f3e = import_ref PackageB//default, inst17 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.89f: %G.assoc_type = import_ref PackageB//default, loc7_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.f03 = import_ref PackageB//default, G, unloaded
// CHECK:STDOUT:   %import_ref.f91 = import_ref PackageB//default, loc13_25, unloaded
// CHECK:STDOUT:   %import_ref.8db: <witness> = import_ref PackageB//default, inst35 [indirect], loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.6a9 = import_ref PackageB//default, inst36 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.4a6: type = import_ref PackageB//default, loc13_14, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.e527c5.1: type = import_ref PackageB//default, loc13_20, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.f77 = import_ref PackageB//default, loc18_25, unloaded
// CHECK:STDOUT:   %import_ref.ccf = import_ref PackageB//default, inst53 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.09b = import_ref PackageB//default, inst54 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.b4f = import_ref PackageB//default, F, unloaded
// CHECK:STDOUT:   %import_ref.3b992b.1: type = import_ref PackageB//default, loc18_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.9e1: type = import_ref PackageB//default, loc18_19, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.f58: <witness> = import_ref PackageB//default, loc23_16, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.3b992b.2: type = import_ref PackageB//default, loc23_6, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.e527c5.2: type = import_ref PackageB//default, loc23_11, loaded [template = constants.%HasG.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestDG = %TestDG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestDG.decl: %TestDG.type = fn_decl @TestDG [template = constants.%TestDG] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %D.ref [template = constants.%D] {
// CHECK:STDOUT:       %PackageB.ref.loc6: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f3e
// CHECK:STDOUT:   .G = imports.%import_ref.89f
// CHECK:STDOUT:   witness = (imports.%import_ref.f03)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.ccf
// CHECK:STDOUT:   .F = imports.%import_ref.09b
// CHECK:STDOUT:   witness = (imports.%import_ref.b4f)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.4a6 as imports.%import_ref.e527c5.1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.3b992b.1 as imports.%import_ref.9e1 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.3b992b.2 as imports.%import_ref.e527c5.2 [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.f58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.cab
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "package_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6a9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDG(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc7: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.9db [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %G.assoc_type = name_ref G, imports.%import_ref.89f [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %G.type.813 = impl_witness_access constants.%impl_witness, element0 [template = constants.%G.e97]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(constants.%Self.102: %HasG.type) [from "package_b.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() [from "package_b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.102) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %H.type.5e7: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %H.891: %H.type.5e7 = struct_value () [template]
// CHECK:STDOUT:   %H.assoc_type: type = assoc_entity_type %Z.type, %H.type.5e7 [template]
// CHECK:STDOUT:   %assoc0: %H.assoc_type = assoc_entity element0, @Z.%H.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%H.decl) [template]
// CHECK:STDOUT:   %H.type.fad: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %H.bea: %H.type.fad = struct_value () [template]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [template = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%H.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %H.decl: %H.type.5e7 = fn_decl @H.1 [template = constants.%H.891] {} {}
// CHECK:STDOUT:   %assoc0: %H.assoc_type = assoc_entity element0, %H.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %assoc0
// CHECK:STDOUT:   witness = (%H.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %H.decl: %H.type.fad = fn_decl @H.2 [template = constants.%H.bea] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %H.type.594: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %H.assoc_type: type = assoc_entity_type %Z.type, %H.type.594 [template]
// CHECK:STDOUT:   %assoc0: %H.assoc_type = assoc_entity element0, imports.%import_ref.8f8 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.a75) [template]
// CHECK:STDOUT:   %H.type.96e: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %H.33c: %H.type.96e = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageAssociatedInterface: <namespace> = namespace file.%PackageAssociatedInterface.import, [template] {
// CHECK:STDOUT:     .Z = %import_ref.0ef
// CHECK:STDOUT:     import PackageAssociatedInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.0ef: type = import_ref PackageAssociatedInterface//default, Z, loaded [template = constants.%Z.type]
// CHECK:STDOUT:   %import_ref.54a = import_ref PackageAssociatedInterface//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.0ab: %H.assoc_type = import_ref PackageAssociatedInterface//default, loc5_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.108 = import_ref PackageAssociatedInterface//default, H, unloaded
// CHECK:STDOUT:   %import_ref.cc2: <witness> = import_ref PackageAssociatedInterface//default, loc8_14, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.e5c: type = import_ref PackageAssociatedInterface//default, loc8_7, loaded [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %import_ref.f0c: type = import_ref PackageAssociatedInterface//default, loc8_12, loaded [template = constants.%Z.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageAssociatedInterface = imports.%PackageAssociatedInterface
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageAssociatedInterface.import = import PackageAssociatedInterface
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.54a
// CHECK:STDOUT:   .H = imports.%import_ref.0ab
// CHECK:STDOUT:   witness = (imports.%import_ref.108)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.e5c as imports.%import_ref.f0c [from "associated_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.cc2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %PackageAssociatedInterface.ref: <namespace> = name_ref PackageAssociatedInterface, imports.%PackageAssociatedInterface [template = imports.%PackageAssociatedInterface]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, imports.%import_ref.0ef [template = constants.%Z.type]
// CHECK:STDOUT:   %H.ref: %H.assoc_type = name_ref H, imports.%import_ref.0ab [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %H.type.594 = impl_witness_access constants.%impl_witness, element0 [template = constants.%H.33c]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H.1(constants.%Self: %Z.type) [from "associated_interface.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2() [from "associated_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [template]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(%T, %X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %Y.type, %K.type [template]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @Y.%K.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AnyParam = %AnyParam.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AnyParam.decl: %AnyParam.type = class_decl @AnyParam [template = constants.%AnyParam.generic] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: @AnyParam.%T.loc4_16.2 (%T) = value_param_pattern %X.patt.loc4_26.1, runtime_param<invalid> [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %X.param: @AnyParam.%T.loc4_16.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = bind_symbolic_name X, 1, %X.param [symbolic = %X.loc4_26.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [template = constants.%Y.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {} {}
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, %K.decl [template = constants.%assoc0]
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
// CHECK:STDOUT:   %X.loc4_26.2: %T = bind_symbolic_name X, 1 [symbolic = %X.loc4_26.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc4_26.2: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%AnyParam
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K(@Y.%Self: %Y.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %X.loc4_26.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc4_26.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.c92: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.c92 = struct_value () [template]
// CHECK:STDOUT:   %GenericInterface.type.714: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.c9a: %GenericInterface.type.714 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [template]
// CHECK:STDOUT:   %AnyParam.generic: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.a277d5.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.422: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.c92, %GenericInterface.generic) [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.ea2: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.069: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.766: %K.type.069 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [template]
// CHECK:STDOUT:   %K.type.7be: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.328: %K.type.7be = struct_value () [template]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %AnyParam.422, %impl_witness [template]
// CHECK:STDOUT:   %L.type: type = fn_type @L [template]
// CHECK:STDOUT:   %L: %L.type = struct_value () [template]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.422 = struct_value () [template]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %Y.type, %K.type.069 [template]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, imports.%import_ref.6d0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [template] {
// CHECK:STDOUT:     .AnyParam = %import_ref.6c9
// CHECK:STDOUT:     .Y = %import_ref.ca2
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6c9: %AnyParam.type = import_ref PackageHasParam//default, AnyParam, loaded [template = constants.%AnyParam.generic]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.8e4 = import_ref PackageHasParam//default, inst34 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ca2: type = import_ref PackageHasParam//default, Y, loaded [template = constants.%Y.type]
// CHECK:STDOUT:   %import_ref.434 = import_ref PackageHasParam//default, inst40 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.991: %K.assoc_type = import_ref PackageHasParam//default, loc7_10, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.c68: %K.type.069 = import_ref PackageHasParam//default, K, loaded [template = constants.%K.766]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.c92 = interface_decl @GenericInterface [template = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %U.patt.loc6_28.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc6_28.1, runtime_param<invalid> [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6_28.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%import_ref.6c9 [template = constants.%AnyParam.generic]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.c92 = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) [template = constants.%AnyParam.422]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_52: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%import_ref.ca2 [template = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [template = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%U.loc6_28.1: type) {
// CHECK:STDOUT:   %U.loc6_28.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_28.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U.loc6_28.2)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.714)]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.714 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.714) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c9a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.434
// CHECK:STDOUT:   .K = imports.%import_ref.991
// CHECK:STDOUT:   witness = (imports.%import_ref.c68)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %K.decl: %K.type.7be = fn_decl @K.2 [template = constants.%K.328] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(constants.%T: type, constants.%X: %T) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.a277d5.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8e4
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.1(constants.%Self.ea2: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %obj.var: ref %AnyParam.422 = var obj
// CHECK:STDOUT:   %obj: ref %AnyParam.422 = bind_name obj, %obj.var
// CHECK:STDOUT:   %.loc13_58.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_58.2: init %AnyParam.422 = class_init (), %obj.var [template = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc13_59: init %AnyParam.422 = converted %.loc13_58.1, %.loc13_58.2 [template = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc13_59
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.422 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%import_ref.ca2 [template = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %K.assoc_type = name_ref K, imports.%import_ref.991 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %K.type.069 = impl_witness_access constants.%impl_witness, element0 [template = constants.%K.328]
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_28.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%U.loc6_28.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.c92, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.c92
// CHECK:STDOUT:   %T.patt => constants.%GenericInterface.type.c92
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %X.patt => constants.%GenericInterface.generic
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self.ea2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Y.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_interface_as_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = fn_type @M [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %M: %M.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.a277d5.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.0da: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.0da = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.954: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.ebd: %GenericInterface.type.954 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %AnyParam.35d: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.0da, %GenericInterface.generic) [template]
// CHECK:STDOUT:   %AnyParam.val: %AnyParam.35d = struct_value () [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.ea2: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.069: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %Y.type, %K.type.069 [template]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, imports.%import_ref.6d0 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.a9a) [template]
// CHECK:STDOUT:   %K.type.d9f: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.4e3: %K.type.d9f = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [template] {
// CHECK:STDOUT:     .AnyParam = %import_ref.6c9
// CHECK:STDOUT:     .Y = %import_ref.ca2
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageGenericInterface: <namespace> = namespace file.%PackageGenericInterface.import, [template] {
// CHECK:STDOUT:     .GenericInterface = %import_ref.012
// CHECK:STDOUT:     import PackageGenericInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref PackageHasParam//default, loc4_34, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.8e4 = import_ref PackageHasParam//default, inst34 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ef8 = import_ref PackageGenericInterface//default, inst28 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ca2: type = import_ref PackageHasParam//default, Y, loaded [template = constants.%Y.type]
// CHECK:STDOUT:   %import_ref.434 = import_ref PackageHasParam//default, inst40 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.991: %K.assoc_type = import_ref PackageHasParam//default, loc7_10, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.6f6 = import_ref PackageHasParam//default, K, unloaded
// CHECK:STDOUT:   %import_ref.e37: <witness> = import_ref PackageGenericInterface//default, loc8_70, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.d44: type = import_ref PackageGenericInterface//default, loc8_47, loaded [template = constants.%AnyParam.35d]
// CHECK:STDOUT:   %import_ref.63d: type = import_ref PackageGenericInterface//default, loc8_67, loaded [template = constants.%Y.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .PackageGenericInterface = imports.%PackageGenericInterface
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %PackageGenericInterface.import = import PackageGenericInterface
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [template = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(constants.%U: type) [from "has_generic_interface.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.954)]
// CHECK:STDOUT:   %Self: %GenericInterface.type.954 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.ebd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.ef8
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "has_param.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.434
// CHECK:STDOUT:   .K = imports.%import_ref.991
// CHECK:STDOUT:   witness = (imports.%import_ref.6f6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.d44 as imports.%import_ref.63d [from "has_generic_interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.e37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(constants.%T: type, constants.%X: %T) [from "has_param.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.a277d5.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8e4
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %obj.var: ref %AnyParam.35d = var obj
// CHECK:STDOUT:   %obj: ref %AnyParam.35d = bind_name obj, %obj.var
// CHECK:STDOUT:   %.loc9_50.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_50.2: init %AnyParam.35d = class_init (), %obj.var [template = constants.%AnyParam.val]
// CHECK:STDOUT:   %.loc9_51: init %AnyParam.35d = converted %.loc9_50.1, %.loc9_50.2 [template = constants.%AnyParam.val]
// CHECK:STDOUT:   assign %obj.var, %.loc9_51
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.35d = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%import_ref.ca2 [template = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %K.assoc_type = name_ref K, imports.%import_ref.991 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %K.type.069 = impl_witness_access constants.%impl_witness, element0 [template = constants.%K.4e3]
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.1(constants.%Self.ea2: %Y.type) [from "has_param.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.2() [from "has_generic_interface.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.0da, constants.%GenericInterface.generic) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.0da
// CHECK:STDOUT:   %T.patt => constants.%GenericInterface.type.0da
// CHECK:STDOUT:   %X => constants.%GenericInterface.generic
// CHECK:STDOUT:   %X.patt => constants.%GenericInterface.generic
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.1(constants.%Self.ea2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [template]
// CHECK:STDOUT:   %Self.a02: %Extra1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [template]
// CHECK:STDOUT:   %Self.f86: %Extra2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [template]
// CHECK:STDOUT:   %Self.9d3: %Extra3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [template]
// CHECK:STDOUT:   %Self.4fb: %Extra4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [template]
// CHECK:STDOUT:   %Self.acd: %Extra5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [template]
// CHECK:STDOUT:   %Self.27e: %Extra6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [template]
// CHECK:STDOUT:   %Self.f8e: %Extra7.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [template]
// CHECK:STDOUT:   %Self.0e4: %Extra8.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.fb7: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.9df: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.478: %F.type.9df = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.9df [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %tuple.type.c53: type = tuple_type (type, type, type, type, type, type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.15d: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [template]
// CHECK:STDOUT:   %C.88d: type = class_type @C, @C(%tuple.type.15d) [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.b71: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.32b: %F.type.b71 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.88d, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %Extra1.decl: type = interface_decl @Extra1 [template = constants.%Extra1.type] {} {}
// CHECK:STDOUT:   %Extra2.decl: type = interface_decl @Extra2 [template = constants.%Extra2.type] {} {}
// CHECK:STDOUT:   %Extra3.decl: type = interface_decl @Extra3 [template = constants.%Extra3.type] {} {}
// CHECK:STDOUT:   %Extra4.decl: type = interface_decl @Extra4 [template = constants.%Extra4.type] {} {}
// CHECK:STDOUT:   %Extra5.decl: type = interface_decl @Extra5 [template = constants.%Extra5.type] {} {}
// CHECK:STDOUT:   %Extra6.decl: type = interface_decl @Extra6 [template = constants.%Extra6.type] {} {}
// CHECK:STDOUT:   %Extra7.decl: type = interface_decl @Extra7 [template = constants.%Extra7.type] {} {}
// CHECK:STDOUT:   %Extra8.decl: type = interface_decl @Extra8 [template = constants.%Extra8.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc13_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_9.1, runtime_param<invalid> [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %Extra1.ref: type = name_ref Extra1, file.%Extra1.decl [template = constants.%Extra1.type]
// CHECK:STDOUT:     %Extra2.ref: type = name_ref Extra2, file.%Extra2.decl [template = constants.%Extra2.type]
// CHECK:STDOUT:     %Extra3.ref: type = name_ref Extra3, file.%Extra3.decl [template = constants.%Extra3.type]
// CHECK:STDOUT:     %Extra4.ref: type = name_ref Extra4, file.%Extra4.decl [template = constants.%Extra4.type]
// CHECK:STDOUT:     %Extra5.ref: type = name_ref Extra5, file.%Extra5.decl [template = constants.%Extra5.type]
// CHECK:STDOUT:     %Extra6.ref: type = name_ref Extra6, file.%Extra6.decl [template = constants.%Extra6.type]
// CHECK:STDOUT:     %Extra7.ref: type = name_ref Extra7, file.%Extra7.decl [template = constants.%Extra7.type]
// CHECK:STDOUT:     %Extra8.ref: type = name_ref Extra8, file.%Extra8.decl [template = constants.%Extra8.type]
// CHECK:STDOUT:     %.loc16_71: %tuple.type.c53 = tuple_literal (%Extra1.ref, %Extra2.ref, %Extra3.ref, %Extra4.ref, %Extra5.ref, %Extra6.ref, %Extra7.ref, %Extra8.ref)
// CHECK:STDOUT:     %.loc16_72: type = converted %.loc16_71, constants.%tuple.type.15d [template = constants.%tuple.type.15d]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.type.15d) [template = constants.%C.88d]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT:   %Self: %Extra1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a02]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT:   %Self: %Extra2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f86]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT:   %Self: %Extra3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9d3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT:   %Self: %Extra4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4fb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT:   %Self: %Extra5.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.acd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT:   %Self: %Extra6.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.27e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT:   %Self: %Extra7.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f8e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT:   %Self: %Extra8.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.0e4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fb7]
// CHECK:STDOUT:   %F.decl: %F.type.9df = fn_decl @F.1 [template = constants.%F.478] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.b71 = fn_decl @F.2 [template = constants.%F.32b] {} {}
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc13_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.fb7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type.15d) {
// CHECK:STDOUT:   %T.loc13_9.2 => constants.%tuple.type.15d
// CHECK:STDOUT:   %T.patt.loc13_9.2 => constants.%tuple.type.15d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.b1d: type = class_type @C, @C(type) [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.ce8: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, imports.%import_ref.911 [template]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [template]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [template]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [template]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [template]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [template]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [template]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [template]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [template]
// CHECK:STDOUT:   %C.d74: type = class_type @C, @C(%tuple.type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %HasExtraInterfaces: <namespace> = namespace file.%HasExtraInterfaces.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.e8a
// CHECK:STDOUT:     .I = %import_ref.9f4
// CHECK:STDOUT:     import HasExtraInterfaces//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.e8a: %C.type = import_ref HasExtraInterfaces//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref HasExtraInterfaces//default, loc13_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.495 = import_ref HasExtraInterfaces//default, inst57 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9f4: type = import_ref HasExtraInterfaces//default, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.f8c = import_ref HasExtraInterfaces//default, inst63 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9b9: %F.assoc_type = import_ref HasExtraInterfaces//default, loc14_21, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.a15 = import_ref HasExtraInterfaces//default, F, unloaded
// CHECK:STDOUT:   %import_ref.92b = import_ref HasExtraInterfaces//default, loc16_79, unloaded
// CHECK:STDOUT:   %import_ref.8b6 = import_ref HasExtraInterfaces//default, inst43 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8c6 = import_ref HasExtraInterfaces//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3dd = import_ref HasExtraInterfaces//default, inst35 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.1a7 = import_ref HasExtraInterfaces//default, inst31 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.de2 = import_ref HasExtraInterfaces//default, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.93a = import_ref HasExtraInterfaces//default, inst23 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.5a8 = import_ref HasExtraInterfaces//default, inst19 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.4be = import_ref HasExtraInterfaces//default, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.2a6: type = import_ref HasExtraInterfaces//default, loc16_72, loaded [template = constants.%C.d74]
// CHECK:STDOUT:   %import_ref.bd0: type = import_ref HasExtraInterfaces//default, loc16_77, loaded [template = constants.%I.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .HasExtraInterfaces = imports.%HasExtraInterfaces
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces.import = import HasExtraInterfaces
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %c.patt: %C.b1d = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.b1d = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.b1d = value_param runtime_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %C [template = constants.%C.b1d] {
// CHECK:STDOUT:       %HasExtraInterfaces.ref.loc5: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [template = imports.%HasExtraInterfaces]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%import_ref.e8a [template = constants.%C.generic]
// CHECK:STDOUT:       %C: type = class_type @C, @C(type) [template = constants.%C.b1d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.b1d = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f8c
// CHECK:STDOUT:   .F = imports.%import_ref.9b9
// CHECK:STDOUT:   witness = (imports.%import_ref.a15)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8b6
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8c6
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3dd
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.1a7
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.de2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.93a
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5a8
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4be
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.2a6 as imports.%import_ref.bd0 [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.92b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%c.param_patt: %C.b1d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.b1d = name_ref c, %c
// CHECK:STDOUT:   %HasExtraInterfaces.ref.loc11: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [template = imports.%HasExtraInterfaces]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%import_ref.9f4 [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, imports.%import_ref.9b9 [template = constants.%assoc0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(constants.%Self.ce8: %I.type) [from "has_extra_interfaces.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ce8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type) {
// CHECK:STDOUT:   %T => constants.%tuple.type
// CHECK:STDOUT:   %T.patt => constants.%tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
