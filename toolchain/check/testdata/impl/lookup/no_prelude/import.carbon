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

// --- use.carbon

import PackageA;
import PackageB;

fn TestCF(c: PackageA.C) {
  c.(PackageA.HasF.F)();
}

fn TestDF(d: PackageB.D) {
  d.(PackageA.HasF.F)();
}

fn TestCG(c: PackageA.C) {
  c.(PackageB.HasG.G)();
}

fn TestDG(d: PackageB.D) {
  d.(PackageB.HasG.G)();
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
  // CHECK:STDERR: fail_use_has_extra_interfaces.carbon:[[@LINE+3]]:3: error: cannot access member of interface `I` in type `C` that does not implement that interface [MissingImplInMemberAccess]
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
// CHECK:STDOUT: --- use.carbon
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
// CHECK:STDOUT:   %Self.1: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %TestDF.type: type = fn_type @TestDF [template]
// CHECK:STDOUT:   %TestDF: %TestDF.type = struct_value () [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.2: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %TestCG.type: type = fn_type @TestCG [template]
// CHECK:STDOUT:   %TestCG: %TestCG.type = struct_value () [template]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.26 [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT:   %TestDG.type: type = fn_type @TestDG [template]
// CHECK:STDOUT:   %TestDG: %TestDG.type = struct_value () [template]
// CHECK:STDOUT:   %G.type.3: type = fn_type @G.3 [template]
// CHECK:STDOUT:   %G.3: %G.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%G.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %PackageA: <namespace> = namespace file.%PackageA.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .HasF = %import_ref.3
// CHECK:STDOUT:     import PackageA//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageB: <namespace> = namespace file.%PackageB.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.11
// CHECK:STDOUT:     .HasG = %import_ref.25
// CHECK:STDOUT:     import PackageB//default
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
// CHECK:STDOUT:   %import_ref.11: type = import_ref PackageB//default, inst+14, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.12 = import_ref PackageB//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref PackageB//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.14: %.8 = import_ref PackageB//default, inst+12, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.15 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref PackageB//default, inst+24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.17: type = import_ref PackageB//default, inst+25, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.18: <witness> = import_ref PackageB//default, inst+30, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.19: type = import_ref PackageB//default, inst+33, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.20: type = import_ref PackageB//default, inst+42, loaded [template = constants.%HasF.type]
// CHECK:STDOUT:   %import_ref.21: <witness> = import_ref PackageB//default, inst+50, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.22: type = import_ref PackageB//default, inst+53, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.23: type = import_ref PackageB//default, inst+54, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.24: <witness> = import_ref PackageB//default, inst+59, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.25: type = import_ref PackageB//default, inst+3, loaded [template = constants.%HasG.type]
// CHECK:STDOUT:   %import_ref.26 = import_ref PackageB//default, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .PackageA = imports.%PackageA
// CHECK:STDOUT:     .PackageB = imports.%PackageB
// CHECK:STDOUT:     .TestCF = %TestCF.decl
// CHECK:STDOUT:     .TestDF = %TestDF.decl
// CHECK:STDOUT:     .TestCG = %TestCG.decl
// CHECK:STDOUT:     .TestDG = %TestDG.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PackageA.import = import PackageA
// CHECK:STDOUT:   %PackageB.import = import PackageB
// CHECK:STDOUT:   %TestCF.decl: %TestCF.type = fn_decl @TestCF [template = constants.%TestCF] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %PackageA.ref.loc5: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDF.decl: %TestDF.type = fn_decl @TestDF [template = constants.%TestDF] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.11 [template = constants.%D]
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestCG.decl: %TestCG.type = fn_decl @TestCG [template = constants.%TestCG] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDG.decl: %TestDG.type = fn_decl @TestDG [template = constants.%TestDG] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %PackageB.ref.loc17: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.11 [template = constants.%D]
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
// CHECK:STDOUT:   .Self = imports.%import_ref.13
// CHECK:STDOUT:   .G = imports.%import_ref.14
// CHECK:STDOUT:   witness = (imports.%import_ref.15)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.19 as imports.%import_ref.20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCF(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageA.ref.loc6: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.3 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc6: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc6()
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
// CHECK:STDOUT: fn @TestDF(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageA.ref: <namespace> = name_ref PackageA, imports.%PackageA [template = imports.%PackageA]
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, imports.%import_ref.3 [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.4 = name_ref F, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10: %F.type.1 = interface_witness_access constants.%.7, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc10()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCG(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %PackageB.ref: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.25 [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %.8 = name_ref G, imports.%import_ref.14 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14: %G.type.1 = interface_witness_access constants.%.10, element0 [template = constants.%G.2]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %.loc14()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(constants.%Self.2: %HasG.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDG(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %PackageB.ref.loc18: <namespace> = name_ref PackageB, imports.%PackageB [template = imports.%PackageB]
// CHECK:STDOUT:   %HasG.ref: type = name_ref HasG, imports.%import_ref.25 [template = constants.%HasG.type]
// CHECK:STDOUT:   %G.ref: %.8 = name_ref G, imports.%import_ref.14 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18: %G.type.1 = interface_witness_access constants.%.11, element0 [template = constants.%G.3]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %.loc18()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.2) {}
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [template]
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
// CHECK:STDOUT:   %.loc16_79: <witness> = interface_witness (%F.decl) [template = constants.%.5]
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
// CHECK:STDOUT: specific @F.1(constants.%C.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_has_extra_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(type) [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.7 [template]
// CHECK:STDOUT:   %Extra8.type: type = facet_type <@Extra8> [template]
// CHECK:STDOUT:   %Self.2: %Extra8.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra7.type: type = facet_type <@Extra7> [template]
// CHECK:STDOUT:   %Self.3: %Extra7.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra6.type: type = facet_type <@Extra6> [template]
// CHECK:STDOUT:   %Self.4: %Extra6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra5.type: type = facet_type <@Extra5> [template]
// CHECK:STDOUT:   %Self.5: %Extra5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra4.type: type = facet_type <@Extra4> [template]
// CHECK:STDOUT:   %Self.6: %Extra4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra3.type: type = facet_type <@Extra3> [template]
// CHECK:STDOUT:   %Self.7: %Extra3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra2.type: type = facet_type <@Extra2> [template]
// CHECK:STDOUT:   %Self.8: %Extra2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Extra1.type: type = facet_type <@Extra1> [template]
// CHECK:STDOUT:   %Self.9: %Extra1.type = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:   %import_ref.7 = import_ref HasExtraInterfaces//default, inst+53, unloaded
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
// CHECK:STDOUT:   %import_ref.18 = import_ref HasExtraInterfaces//default, inst+79, unloaded
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
