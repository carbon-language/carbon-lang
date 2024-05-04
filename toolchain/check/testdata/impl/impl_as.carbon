// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn F();
}

class C {
  impl as Simple {
    fn F() {
      // C is a complete type here.
      var c: C = {};
    }
  }
}

// CHECK:STDOUT: --- impl_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Simple [template]
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Simple, F [template]
// CHECK:STDOUT:   %.4: <associated F in Simple> = assoc_entity element0, @Simple.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%struct.2) [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc8: <associated F in Simple> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Simple {
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
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_19.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc15_19.2: init C = class_init (), %c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc15_20: init C = converted %.loc15_19.1, %.loc15_19.2 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %c.var, %.loc15_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
