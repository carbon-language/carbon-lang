// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C;

interface I {
  // TODO: Use `Self` once we perform substitution for it.
  fn F[self: C]() -> i32;
}

class C {
  extend impl as I {
    fn F[self: C]() -> i32;
  }
}

fn F(c: C) -> i32 {
  return c.F();
}

// CHECK:STDOUT: --- instance_method.carbon
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc7
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc7: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl.loc14: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.3 [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl.loc7 [template = constants.%C]
// CHECK:STDOUT:     %c.loc20_6.1: C = param c
// CHECK:STDOUT:     @F.3.%c: C = bind_name c, %c.loc20_6.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc7 [template = constants.%C]
// CHECK:STDOUT:     %self.loc11_8.1: C = param self
// CHECK:STDOUT:     %self.loc11_8.2: C = bind_name self, %self.loc11_8.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc11
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc7 [template = constants.%C]
// CHECK:STDOUT:     %self.loc16_10.1: C = param self
// CHECK:STDOUT:     %self.loc16_10.2: C = bind_name self, %self.loc16_10.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@I.%self.loc11_8.2: C]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@impl.%self.loc16_10.2: C]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%c: C) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %F.ref: <associated <function> in I> = name_ref F, @I.%.loc11 [template = constants.%.3]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%F]
// CHECK:STDOUT:   %.loc21_11: <bound method> = bound_method %c.ref, %.1
// CHECK:STDOUT:   %.loc21_13.1: init i32 = call %.loc21_11(%c.ref)
// CHECK:STDOUT:   %.loc21_15: i32 = value_of_initializer %.loc21_13.1
// CHECK:STDOUT:   %.loc21_13.2: i32 = converted %.loc21_13.1, %.loc21_15
// CHECK:STDOUT:   return %.loc21_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
