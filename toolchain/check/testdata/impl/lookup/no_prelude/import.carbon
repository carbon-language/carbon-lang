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

// CHECK:STDOUT: --- package_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = interface_type @HasF [template]
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
// CHECK:STDOUT:   %HasG.type: type = interface_type @HasG [template]
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
// CHECK:STDOUT:   %HasF.type: type = interface_type @HasF [template]
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
// CHECK:STDOUT:   %HasF.type: type = interface_type @HasF [template]
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
// CHECK:STDOUT:   %HasG.type: type = interface_type @HasG [template]
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
