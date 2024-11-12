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
  // CHECK:STDERR: use_generic_interface_as_param.carbon:[[@LINE+3]]:3: error: cannot access member of interface `Y` in type `AnyParam` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   obj.(PackageHasParam.Y.K)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  obj.(PackageHasParam.Y.K)();
}

// CHECK:STDOUT: --- package_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [template]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc11: <witness> = interface_witness (%F.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT: specific @F.1(constants.%C) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasG.%G.decl [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.2: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.1) [template]
// CHECK:STDOUT:   %G.type.3: type = fn_type @G.3 [template]
// CHECK:STDOUT:   %G.3: %G.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%G.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .HasF = %import_ref.3
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageA//default, inst+12, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageA//default, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageA//default, inst+1, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageA//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref PackageA//default, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.6: %F.type.2 = import_ref PackageA//default, inst+5, loaded [template = constants.%F.2]
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
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, imports.%import_ref.3 [template = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {} {}
// CHECK:STDOUT:   %.loc7: %.1 = assoc_entity element0, %G.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %.loc7
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .F = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {} {}
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness (%G.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc18: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.3 = fn_decl @G.3 [template = constants.%G.3] {} {}
// CHECK:STDOUT:   %.loc23: <witness> = interface_witness (%G.decl) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
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
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(constants.%Self.2: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%C) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%D) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%D) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .HasF = %import_ref.3
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageA//default, inst+12, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageA//default, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageA//default, inst+1, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageA//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref PackageA//default, inst+10, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref PackageA//default, inst+23, loaded [template = constants.%.6]
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
// CHECK:STDOUT:     %PackageA.ref.loc6: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .F = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc7: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.3 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc7()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_df.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.1: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.2: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .HasF = %import_ref.3
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.1
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageB//default, inst+14, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageB//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageA//default, inst+1, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageA//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref PackageA//default, inst+10, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageA//default, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.10: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.11 = import_ref PackageA//default, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref PackageB//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageB//default, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.16: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+33, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+42, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.20: <witness> = import_ref PackageB//default, inst+50, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.21: type = import_ref PackageB//default, inst+53, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.22: type = import_ref PackageB//default, inst+54, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.23 = import_ref PackageB//default, inst+59, unloaded
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
// CHECK:STDOUT:     %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.1 [template = constants.%D]
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .F = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.12
// CHECK:STDOUT:   .G = imports.%import_ref.13
// CHECK:STDOUT:   witness = (imports.%import_ref.14)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.18 as imports.%import_ref.19 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDF(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.3 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc8()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self.1: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.2: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .HasG = %import_ref.3
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageA//default, inst+12, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageA//default, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageB//default, inst+3, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageB//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref PackageB//default, inst+12, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageA//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref PackageA//default, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.12: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageA//default, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.16: <witness> = import_ref PackageB//default, inst+30, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+33, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+42, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.20 = import_ref PackageB//default, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref PackageB//default, inst+53, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.22: type = import_ref PackageB//default, inst+54, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.23 = import_ref PackageB//default, inst+59, unloaded
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
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .G = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   .F = imports.%import_ref.9
// CHECK:STDOUT:   witness = (imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.18 as imports.%import_ref.19 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCG(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.3 [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %.4 = name_ref G, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8: %G.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%G.2]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %.loc8()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(constants.%Self.1: %HasG.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_dg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.2: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.1
// CHECK:STDOUT:     .HasG = %import_ref.3
// CHECK:STDOUT:     import PackageB//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageB//default, inst+14, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageB//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageB//default, inst+3, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageB//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref PackageB//default, inst+12, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageB//default, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.10: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.11 = import_ref PackageB//default, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref PackageB//default, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageB//default, inst+40, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref PackageB//default, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+33, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.16: type = import_ref PackageB//default, inst+42, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+53, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+54, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.20: <witness> = import_ref PackageB//default, inst+59, loaded [template = constants.%.6]
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
// CHECK:STDOUT:     %PackageB.ref.loc6: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.1 [template = constants.%D]
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .G = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.12
// CHECK:STDOUT:   .F = imports.%import_ref.13
// CHECK:STDOUT:   witness = (imports.%import_ref.14)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.18 as imports.%import_ref.19 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDG(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc7: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.3 [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %.4 = name_ref G, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7: %G.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%G.2]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %.loc7()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(constants.%Self.1: %HasG.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %H.type.1: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %H.1: %H.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Z.type, %H.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Z.%H.decl [template]
// CHECK:STDOUT:   %H.type.2: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %H.2: %H.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%H.2) [template]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %H.decl: %H.type.1 = fn_decl @H.1 [template = constants.%H.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %H.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %.loc5
// CHECK:STDOUT:   witness = (%H.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %H.decl: %H.type.2 = fn_decl @H.2 [template = constants.%H.2] {} {}
// CHECK:STDOUT:   %.loc8_14: <witness> = interface_witness (%H.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   witness = %.loc8_14
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
// CHECK:STDOUT: specific @H.1(constants.%empty_tuple.type) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_associated_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %H.type.1: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %H.1: %H.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Z.type, %H.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %H.type.2: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %H.2: %H.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%H.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageAssociatedInterface: <namespace> = namespace file.%PackageAssociatedInterface.import, [template] {
// CHECK:STDOUT:     .Z = %import_ref.1
// CHECK:STDOUT:     import PackageAssociatedInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref PackageAssociatedInterface//default, inst+1, loaded [template = constants.%Z.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageAssociatedInterface//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %.1 = import_ref PackageAssociatedInterface//default, inst+10, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageAssociatedInterface//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref PackageAssociatedInterface//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.6: type = import_ref PackageAssociatedInterface//default, inst+13, loaded [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %import_ref.7: type = import_ref PackageAssociatedInterface//default, inst+14, loaded [template = constants.%Z.type]
// CHECK:STDOUT:   %import_ref.8: <witness> = import_ref PackageAssociatedInterface//default, inst+19, loaded [template = constants.%.3]
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
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .H = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.6 as imports.%import_ref.7 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %PackageAssociatedInterface.ref: <namespace> = name_ref PackageAssociatedInterface, imports.%PackageAssociatedInterface [template = imports.%PackageAssociatedInterface]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, imports.%import_ref.1 [template = constants.%Z.type]
// CHECK:STDOUT:   %H.ref: %.1 = name_ref H, imports.%import_ref.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_5: %H.type.1 = interface_witness_access constants.%.3, element0 [template = constants.%H.2]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %.loc7_5()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H.1(constants.%Self: %Z.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2();
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %AnyParam.1: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %AnyParam.2: type = class_type @AnyParam, @AnyParam(%T, %X) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Y.type, %K.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Y.%K.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AnyParam = %AnyParam.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AnyParam.decl: %AnyParam.type = class_decl @AnyParam [template = constants.%AnyParam.1] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %X.patt.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: @AnyParam.%T.loc4_16.2 (%T) = value_param_pattern %X.patt.loc4_26.1, runtime_param<invalid> [symbolic = %X.patt.loc4_26.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %X.param: @AnyParam.%T.loc4_16.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %X.loc4_26.1: @AnyParam.%T.loc4_16.2 (%T) = bind_symbolic_name X, 1, %X.param [symbolic = %X.loc4_26.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [template = constants.%Y.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {} {}
// CHECK:STDOUT:   %.loc7: %.3 = assoc_entity element0, %K.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %.loc7
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
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%AnyParam.2
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
// CHECK:STDOUT:   %GenericInterface.type.1: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type.1 = struct_value () [template]
// CHECK:STDOUT:   %GenericInterface.type.2: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.1: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [template]
// CHECK:STDOUT:   %AnyParam.1: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X.1: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.2: type = class_type @AnyParam, @AnyParam(%T, %X.1) [symbolic]
// CHECK:STDOUT:   %X.2: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.1: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.3: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.1, %GenericInterface) [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.2: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.1: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.1: %K.type.1 = struct_value () [template]
// CHECK:STDOUT:   %K.type.2: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.2: %K.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%K.1) [template]
// CHECK:STDOUT:   %L.type: type = fn_type @L [template]
// CHECK:STDOUT:   %L: %L.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %AnyParam.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Y.type, %K.type.2 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [template] {
// CHECK:STDOUT:     .AnyParam = %import_ref.1
// CHECK:STDOUT:     .Y = %import_ref.3
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %AnyParam.type = import_ref PackageHasParam//default, inst+14, loaded [template = constants.%AnyParam.1]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageHasParam//default, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref PackageHasParam//default, inst+26, loaded [template = constants.%Y.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageHasParam//default, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.5: %.5 = import_ref PackageHasParam//default, inst+34, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.6: %K.type.2 = import_ref PackageHasParam//default, inst+30, loaded [template = constants.%K.2]
// CHECK:STDOUT:   %import_ref.7 = import_ref PackageHasParam//default, inst+30, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageHasParam = imports.%PackageHasParam
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageHasParam.import = import PackageHasParam
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.1 = interface_decl @GenericInterface [template = constants.%GenericInterface] {
// CHECK:STDOUT:     %U.patt.loc6_28.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc6_28.1, runtime_param<invalid> [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6_28.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_6: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:     %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%import_ref.1 [template = constants.%AnyParam.1]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.1 = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface]
// CHECK:STDOUT:     %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.1, constants.%GenericInterface) [template = constants.%AnyParam.3]
// CHECK:STDOUT:     %PackageHasParam.ref.loc8_52: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, imports.%import_ref.3 [template = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [template = constants.%L] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%U.loc6_28.1: type) {
// CHECK:STDOUT:   %U.loc6_28.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_28.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_28.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_28.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U.loc6_28.2)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .K = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %AnyParam as %Y.ref {
// CHECK:STDOUT:   %K.decl: %K.type.1 = fn_decl @K.1 [template = constants.%K.1] {} {}
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%K.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(constants.%T: type, constants.%X.2: @AnyParam.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic = %X (constants.%X.1)]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.2(constants.%Self.2: %Y.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%import_ref.1 [template = constants.%AnyParam.1]
// CHECK:STDOUT:   %GenericInterface.ref: %GenericInterface.type.1 = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.1, constants.%GenericInterface) [template = constants.%AnyParam.3]
// CHECK:STDOUT:   %obj.var: ref %AnyParam.3 = var obj
// CHECK:STDOUT:   %obj: ref %AnyParam.3 = bind_name obj, %obj.var
// CHECK:STDOUT:   %.loc13_58.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_58.2: init %AnyParam.3 = class_init (), %obj.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_59: init %AnyParam.3 = converted %.loc13_58.1, %.loc13_58.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %obj.var, %.loc13_59
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.3 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%import_ref.3 [template = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %.5 = name_ref K, imports.%import_ref.5 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14: %K.type.2 = interface_witness_access constants.%.3, element0 [template = constants.%K.1]
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %.loc14()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_28.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@GenericInterface.%U.loc6_28.2) {
// CHECK:STDOUT:   %U.loc6_28.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_28.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X.1
// CHECK:STDOUT:   %X.patt => constants.%X.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X.2
// CHECK:STDOUT:   %X.patt => constants.%X.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.1, constants.%GenericInterface) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.1
// CHECK:STDOUT:   %T.patt => constants.%GenericInterface.type.1
// CHECK:STDOUT:   %X => constants.%GenericInterface
// CHECK:STDOUT:   %X.patt => constants.%GenericInterface
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.2(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.2(constants.%AnyParam.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_interface_as_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = fn_type @M [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %M: %M.type = struct_value () [template]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [template]
// CHECK:STDOUT:   %AnyParam.1: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X.1: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.2: type = class_type @AnyParam, @AnyParam(%T, %X.1) [symbolic]
// CHECK:STDOUT:   %X.2: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt.1: @AnyParam.%T (%T) = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.1: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type.1 = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.2: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.3: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.1, %GenericInterface) [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %AnyParam.3 = struct_value () [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.3: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Y.type, %K.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageHasParam: <namespace> = namespace file.%PackageHasParam.import, [template] {
// CHECK:STDOUT:     .AnyParam = %import_ref.1
// CHECK:STDOUT:     .Y = %import_ref.5
// CHECK:STDOUT:     import PackageHasParam//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageGenericInterface: <namespace> = namespace file.%PackageGenericInterface.import, [template] {
// CHECK:STDOUT:     .GenericInterface = %import_ref.3
// CHECK:STDOUT:     import PackageGenericInterface//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %AnyParam.type = import_ref PackageHasParam//default, inst+14, loaded [template = constants.%AnyParam.1]
// CHECK:STDOUT:   %import_ref.2 = import_ref PackageHasParam//default, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.3: %GenericInterface.type.1 = import_ref PackageGenericInterface//default, inst+9, loaded [template = constants.%GenericInterface]
// CHECK:STDOUT:   %import_ref.4 = import_ref PackageGenericInterface//default, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.5: type = import_ref PackageHasParam//default, inst+26, loaded [template = constants.%Y.type]
// CHECK:STDOUT:   %import_ref.6 = import_ref PackageHasParam//default, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.7: %.4 = import_ref PackageHasParam//default, inst+34, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageHasParam//default, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref PackageHasParam//default, inst+30, unloaded
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
// CHECK:STDOUT: generic interface @GenericInterface(constants.%U: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:   %Self: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   .K = imports.%import_ref.7
// CHECK:STDOUT:   witness = (imports.%import_ref.8)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @AnyParam(constants.%T: type, constants.%X.2: @AnyParam.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %X: %T = bind_symbolic_name X, 1 [symbolic = %X (constants.%X.1)]
// CHECK:STDOUT:   %X.patt: %T = symbolic_binding_pattern X, 1 [symbolic = %X.patt (constants.%X.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PackageHasParam.ref.loc8: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %AnyParam.ref: %AnyParam.type = name_ref AnyParam, imports.%import_ref.1 [template = constants.%AnyParam.1]
// CHECK:STDOUT:   %PackageGenericInterface.ref: <namespace> = name_ref PackageGenericInterface, imports.%PackageGenericInterface [template = imports.%PackageGenericInterface]
// CHECK:STDOUT:   %GenericInterface.ref: %GenericInterface.type.1 = name_ref GenericInterface, imports.%import_ref.3 [template = constants.%GenericInterface]
// CHECK:STDOUT:   %AnyParam: type = class_type @AnyParam, @AnyParam(constants.%GenericInterface.type.1, constants.%GenericInterface) [template = constants.%AnyParam.3]
// CHECK:STDOUT:   %obj.var: ref %AnyParam.3 = var obj
// CHECK:STDOUT:   %obj: ref %AnyParam.3 = bind_name obj, %obj.var
// CHECK:STDOUT:   %.loc9_50.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_50.2: init %AnyParam.3 = class_init (), %obj.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_51: init %AnyParam.3 = converted %.loc9_50.1, %.loc9_50.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %obj.var, %.loc9_51
// CHECK:STDOUT:   %obj.ref: ref %AnyParam.3 = name_ref obj, %obj
// CHECK:STDOUT:   %PackageHasParam.ref.loc13: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, imports.%import_ref.5 [template = constants.%Y.type]
// CHECK:STDOUT:   %K.ref: %.4 = name_ref K, imports.%import_ref.7 [template = constants.%.5]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K(constants.%Self.3: %Y.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X.1
// CHECK:STDOUT:   %X.patt => constants.%X.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%T, constants.%X.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %X => constants.%X.2
// CHECK:STDOUT:   %X.patt => constants.%X.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@GenericInterface.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AnyParam(constants.%GenericInterface.type.1, constants.%GenericInterface) {
// CHECK:STDOUT:   %T => constants.%GenericInterface.type.1
// CHECK:STDOUT:   %T.patt => constants.%GenericInterface.type.1
// CHECK:STDOUT:   %X => constants.%GenericInterface
// CHECK:STDOUT:   %X.patt => constants.%GenericInterface
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self.3) {}
// CHECK:STDOUT:
