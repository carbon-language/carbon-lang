// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface HasF {
  fn F();
}

class C {
  extend impl as HasF {
    fn F() {}
  }
}

fn G(c: C) {
  C.F();
  c.F();
}

// CHECK:STDOUT: --- extend_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @HasF [template]
// CHECK:STDOUT:   %Self: HasF = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @HasF, F [template]
// CHECK:STDOUT:   %.4: <associated F in HasF> = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%struct.2) [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc17_6.1: C = param c
// CHECK:STDOUT:     @G.%c: C = bind_name c, %c.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: HasF = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc8: <associated F in HasF> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as HasF {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %F.ref.loc18: <associated F in HasF> = name_ref F, @HasF.%.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.1: F = interface_witness_access @impl.%.1, element0 [template = constants.%struct.2]
// CHECK:STDOUT:   %F.call.loc18: init () = call %.1()
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc19: <associated F in HasF> = name_ref F, @HasF.%.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.2: F = interface_witness_access @impl.%.1, element0 [template = constants.%struct.2]
// CHECK:STDOUT:   %F.call.loc19: init () = call %.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
