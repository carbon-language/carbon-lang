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

// --- fail_todo_use_generic_interface_as_param.carbon

library "[[@TEST_NAME]]";

import PackageHasParam;
import PackageGenericInterface;

fn M() {
  var obj: PackageHasParam.AnyParam(
    PackageGenericInterface.GenericInterface) = {};
  // CHECK:STDERR: fail_todo_use_generic_interface_as_param.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Y` in type `AnyParam(<cannot stringify inst+41 kind StructValue>)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   obj.(PackageHasParam.Y.K)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
  // CHECK:STDERR: fail_use_has_extra_interfaces.carbon:[[@LINE+3]]:3: error: cannot access member of interface `I` in type `C(type)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c.(HasExtraInterfaces.I.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  c.(HasExtraInterfaces.I.F)();
}

// CHECK:STDOUT: --- package_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: %HasF.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
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
// CHECK:STDOUT:   %.loc11: <witness> = interface_witness (%F.decl) [template = constants.%.6]
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
// CHECK:STDOUT: specific @F.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasG.%G.decl [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: %HasG.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.2: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: %HasF.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.1) [template]
// CHECK:STDOUT:   %G.type.3: type = fn_type @G.3 [template]
// CHECK:STDOUT:   %G.3: %G.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.9: %HasG.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%G.3) [template]
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
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness (%G.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %D.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc18: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %D.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.3 = fn_decl @G.3 [template = constants.%G.3] {} {}
// CHECK:STDOUT:   %.loc23: <witness> = interface_witness (%G.decl) [template = constants.%.10]
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
// CHECK:STDOUT: specific @G.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%.9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_cf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %TestCF.type: type = fn_type @TestCF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCF: %TestCF.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
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
// CHECK:STDOUT:   %import_ref.8: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref PackageA//default, inst+24, loaded [template = constants.%.6]
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
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.1: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
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
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageA//default, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.10: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.11 = import_ref PackageA//default, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref PackageB//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageB//default, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.16: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+34, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+43, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.20: <witness> = import_ref PackageB//default, inst+52, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.21: type = import_ref PackageB//default, inst+55, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.22: type = import_ref PackageB//default, inst+56, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.23 = import_ref PackageB//default, inst+62, unloaded
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
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
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
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageA//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref PackageA//default, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref PackageA//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref PackageA//default, inst+17, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.12: type = import_ref PackageA//default, inst+18, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageA//default, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.16: <witness> = import_ref PackageB//default, inst+31, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+34, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+43, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.20 = import_ref PackageB//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref PackageB//default, inst+55, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.22: type = import_ref PackageB//default, inst+56, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.23 = import_ref PackageB//default, inst+62, unloaded
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
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.1: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
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
// CHECK:STDOUT:   %import_ref.8 = import_ref PackageB//default, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.10: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.11 = import_ref PackageB//default, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref PackageB//default, inst+40, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageB//default, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref PackageB//default, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref PackageB//default, inst+34, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.16: type = import_ref PackageB//default, inst+43, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.17 = import_ref PackageB//default, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref PackageB//default, inst+55, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+56, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.20: <witness> = import_ref PackageB//default, inst+62, loaded [template = constants.%.6]
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
// CHECK:STDOUT:   %.3: %Z.type = facet_value %empty_tuple.type, %empty_tuple.type [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%H.2) [template]
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
// CHECK:STDOUT:   %.loc8_14: <witness> = interface_witness (%H.decl) [template = constants.%.4]
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
// CHECK:STDOUT: specific @H.1(constants.%.3) {}
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
// CHECK:STDOUT:   %import_ref.6: type = import_ref PackageAssociatedInterface//default, inst+13, loaded [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %import_ref.7: type = import_ref PackageAssociatedInterface//default, inst+14, loaded [template = constants.%Z.type]
// CHECK:STDOUT:   %import_ref.8: <witness> = import_ref PackageAssociatedInterface//default, inst+20, loaded [template = constants.%.3]
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X.1: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.2: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.3: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.1, %GenericInterface) [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.2: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.1: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.1: %K.type.1 = struct_value () [template]
// CHECK:STDOUT:   %K.type.2: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.2: %K.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: %Y.type = facet_value %AnyParam.3, %AnyParam.3 [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%K.1) [template]
// CHECK:STDOUT:   %L.type: type = fn_type @L [template]
// CHECK:STDOUT:   %L: %L.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %AnyParam.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %Y.type, %K.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.7 [template]
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
// CHECK:STDOUT:   %import_ref.5: %.6 = import_ref PackageHasParam//default, inst+34, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.6: %K.type.2 = import_ref PackageHasParam//default, inst+30, loaded [template = constants.%K.2]
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
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness (%K.decl) [template = constants.%.4]
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
// CHECK:STDOUT:   %K.ref: %.6 = name_ref K, imports.%import_ref.5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14: %K.type.2 = interface_witness_access constants.%.4, element0 [template = constants.%K.1]
// CHECK:STDOUT:   %K.call: init %empty_tuple.type = call %.loc14()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%U) {
// CHECK:STDOUT:   %U.loc6_28.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_28.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%U.loc6_28.2) {
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
// CHECK:STDOUT: specific @K.2(constants.%.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_generic_interface_as_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = fn_type @M [template]
// CHECK:STDOUT:   %M: %M.type = struct_value () [template]
// CHECK:STDOUT:   %AnyParam.type: type = generic_class_type @AnyParam [template]
// CHECK:STDOUT:   %AnyParam.1: %AnyParam.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %X.1: %T = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.2: @AnyParam.%T (%T) = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %X.patt.2: %T = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.1: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type.1 = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.2: type = facet_type <@GenericInterface, @GenericInterface(%U)> [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AnyParam.3: type = class_type @AnyParam, @AnyParam(%GenericInterface.type.1, %GenericInterface) [template]
// CHECK:STDOUT:   %struct: %AnyParam.3 = struct_value () [template]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [template]
// CHECK:STDOUT:   %Self.3: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
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
// CHECK:STDOUT:   %PackageHasParam.ref.loc14: <namespace> = name_ref PackageHasParam, imports.%PackageHasParam [template = imports.%PackageHasParam]
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
// CHECK:STDOUT: specific @GenericInterface(%U) {
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
// CHECK:STDOUT: --- has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [template]
// CHECK:STDOUT:   %Self.1: %Extra1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [template]
// CHECK:STDOUT:   %Self.2: %Extra2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [template]
// CHECK:STDOUT:   %Self.3: %Extra3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [template]
// CHECK:STDOUT:   %Self.4: %Extra4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [template]
// CHECK:STDOUT:   %Self.5: %Extra5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [template]
// CHECK:STDOUT:   %Self.6: %Extra6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [template]
// CHECK:STDOUT:   %Self.7: %Extra7.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [template]
// CHECK:STDOUT:   %Self.8: %Extra8.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.9: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type, type, type, type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%tuple.type.2) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: %I.type = facet_value %C.3, %C.3 [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc13_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_9.1, runtime_param<invalid> [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %Extra1.ref: type = name_ref Extra1, file.%Extra1.decl [template = constants.%Extra1.type]
// CHECK:STDOUT:     %Extra2.ref: type = name_ref Extra2, file.%Extra2.decl [template = constants.%Extra2.type]
// CHECK:STDOUT:     %Extra3.ref: type = name_ref Extra3, file.%Extra3.decl [template = constants.%Extra3.type]
// CHECK:STDOUT:     %Extra4.ref: type = name_ref Extra4, file.%Extra4.decl [template = constants.%Extra4.type]
// CHECK:STDOUT:     %Extra5.ref: type = name_ref Extra5, file.%Extra5.decl [template = constants.%Extra5.type]
// CHECK:STDOUT:     %Extra6.ref: type = name_ref Extra6, file.%Extra6.decl [template = constants.%Extra6.type]
// CHECK:STDOUT:     %Extra7.ref: type = name_ref Extra7, file.%Extra7.decl [template = constants.%Extra7.type]
// CHECK:STDOUT:     %Extra8.ref: type = name_ref Extra8, file.%Extra8.decl [template = constants.%Extra8.type]
// CHECK:STDOUT:     %.loc16_71: %tuple.type.1 = tuple_literal (%Extra1.ref, %Extra2.ref, %Extra3.ref, %Extra4.ref, %Extra5.ref, %Extra6.ref, %Extra7.ref, %Extra8.ref)
// CHECK:STDOUT:     %.loc16_7: type = converted %.loc16_71, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.type.2) [template = constants.%C.3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT:   %Self: %Extra1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT:   %Self: %Extra2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT:   %Self: %Extra3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT:   %Self: %Extra4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT:   %Self: %Extra5.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT:   %Self: %Extra6.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT:   %Self: %Extra7.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT:   %Self: %Extra8.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc14: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc16_79: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc16_79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc13_9.1: type) {
// CHECK:STDOUT:   %T.loc13_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc13: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
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
// CHECK:STDOUT: specific @F.1(constants.%Self.9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type.2) {
// CHECK:STDOUT:   %T.loc13_9.2 => constants.%tuple.type.2
// CHECK:STDOUT:   %T.patt.loc13_9.2 => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(type) [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [template]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [template]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [template]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [template]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [template]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [template]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [template]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Extra1.type, %Extra2.type, %Extra3.type, %Extra4.type, %Extra5.type, %Extra6.type, %Extra7.type, %Extra8.type) [template]
// CHECK:STDOUT:   %C.4: type = class_type @C, @C(%tuple.type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %HasExtraInterfaces: <namespace> = namespace file.%HasExtraInterfaces.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .I = %import_ref.3
// CHECK:STDOUT:     import HasExtraInterfaces//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref HasExtraInterfaces//default, inst+39, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.2 = import_ref HasExtraInterfaces//default, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref HasExtraInterfaces//default, inst+49, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.4 = import_ref HasExtraInterfaces//default, inst+51, unloaded
// CHECK:STDOUT:   %import_ref.5: %.4 = import_ref HasExtraInterfaces//default, inst+57, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref HasExtraInterfaces//default, inst+53, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref HasExtraInterfaces//default, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref HasExtraInterfaces//default, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref HasExtraInterfaces//default, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref HasExtraInterfaces//default, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref HasExtraInterfaces//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref HasExtraInterfaces//default, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref HasExtraInterfaces//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref HasExtraInterfaces//default, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref HasExtraInterfaces//default, inst+72, loaded [template = constants.%C.4]
// CHECK:STDOUT:   %import_ref.17: type = import_ref HasExtraInterfaces//default, inst+74, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.18 = import_ref HasExtraInterfaces//default, inst+80, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .HasExtraInterfaces = imports.%HasExtraInterfaces
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasExtraInterfaces.import = import HasExtraInterfaces
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %c.patt: %C.3 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.3 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %HasExtraInterfaces.ref.loc5: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [template = imports.%HasExtraInterfaces]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%import_ref.1 [template = constants.%C.1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(type) [template = constants.%C.3]
// CHECK:STDOUT:     %c.param: %C.3 = value_param runtime_param0
// CHECK:STDOUT:     %c: %C.3 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .F = imports.%import_ref.5
// CHECK:STDOUT:   witness = (imports.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra8 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra7 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra6 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.10
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra5 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.11
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.12
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra3 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.13
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.14
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Extra1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.15
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%c.param_patt: %C.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.3 = name_ref c, %c
// CHECK:STDOUT:   %HasExtraInterfaces.ref.loc11: <namespace> = name_ref HasExtraInterfaces, imports.%HasExtraInterfaces [template = imports.%HasExtraInterfaces]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%import_ref.3 [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(constants.%Self.1: %I.type) {
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
// CHECK:STDOUT: specific @F(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.type) {
// CHECK:STDOUT:   %T => constants.%tuple.type
// CHECK:STDOUT:   %T.patt => constants.%tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
