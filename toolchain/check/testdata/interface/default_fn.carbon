// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  interface I {
    // TODO: Use `default` here.
    fn F() {
      // I and F are both complete here, and the impl below is in scope.
      var c: C = {};
      c.(I.F)();
    }
  }

  impl C as I {
    fn F() {}
  }
}

// CHECK:STDOUT: --- default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in I> = assoc_entity element0, @I.%F [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (@impl.%F) [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT:   %.8: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %.loc10: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc10
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc12_19.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc12_19.2: init C = class_init (), %c.var [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_20: init C = converted %.loc12_19.1, %.loc12_19.2 [template = constants.%.8]
// CHECK:STDOUT:   assign %c.var, %.loc12_20
// CHECK:STDOUT:   %c.ref: ref C = name_ref c, %c
// CHECK:STDOUT:   %I.ref: type = name_ref I, @C.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated <function> in I> = name_ref F, @I.%.loc10 [template = constants.%.3]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%F]
// CHECK:STDOUT:   %F.call: init () = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
