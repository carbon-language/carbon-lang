// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn F();
  fn G[self: Self]();
}

impl i32 as Simple {
  fn F();
  fn G[self: i32]();
}

fn NonInstanceCall(n: i32) {
  n.(Simple.F)();
}

fn InstanceCall(n: i32) {
  n.(Simple.G)();
}

fn NonInstanceCallIndirect(p: i32*) {
  p->(Simple.F)();
}

fn InstanceCallIndirect(p: i32*) {
  p->(Simple.G)();
}

// CHECK:STDOUT: --- compound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Simple [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Simple, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Simple> = assoc_entity element0, @Simple.%F [template]
// CHECK:STDOUT:   %.4: <associated <function> in Simple> = assoc_entity element1, @Simple.%G [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (@impl.%F, @impl.%G) [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %.7: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .NonInstanceCall = %NonInstanceCall
// CHECK:STDOUT:     .InstanceCall = %InstanceCall
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect
// CHECK:STDOUT:     .InstanceCallIndirect = %InstanceCallIndirect
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, %Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCall: <function> = fn_decl @NonInstanceCall [template] {
// CHECK:STDOUT:     %n.loc17_20.1: i32 = param n
// CHECK:STDOUT:     @NonInstanceCall.%n: i32 = bind_name n, %n.loc17_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall: <function> = fn_decl @InstanceCall [template] {
// CHECK:STDOUT:     %n.loc21_17.1: i32 = param n
// CHECK:STDOUT:     @InstanceCall.%n: i32 = bind_name n, %n.loc21_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCallIndirect: <function> = fn_decl @NonInstanceCallIndirect [template] {
// CHECK:STDOUT:     %.loc25: type = ptr_type i32 [template = constants.%.7]
// CHECK:STDOUT:     %p.loc25_28.1: i32* = param p
// CHECK:STDOUT:     @NonInstanceCallIndirect.%p: i32* = bind_name p, %p.loc25_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect: <function> = fn_decl @InstanceCallIndirect [template] {
// CHECK:STDOUT:     %.loc29: type = ptr_type i32 [template = constants.%.7]
// CHECK:STDOUT:     %p.loc29_25.1: i32* = param p
// CHECK:STDOUT:     @InstanceCallIndirect.%p: i32* = bind_name p, %p.loc29_25.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %.loc8: <associated <function> in Simple> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:   %G: <function> = fn_decl @G.1 [template] {
// CHECK:STDOUT:     %Self.ref: Simple = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_14.1: type = facet_type_access %Self.ref [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_14.2: type = converted %Self.ref, %.loc9_14.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc9_8.1: Self = param self
// CHECK:STDOUT:     %self.loc9_8.2: Self = bind_name self, %self.loc9_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_21: <associated <function> in Simple> = assoc_entity element1, %G [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   .G = %.loc9_21
// CHECK:STDOUT:   witness = (%F, %G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as Simple {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G.2 [template] {
// CHECK:STDOUT:     %self.loc14_8.1: i32 = param self
// CHECK:STDOUT:     %self.loc14_8.2: i32 = bind_name self, %self.loc14_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F, %G) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[@Simple.%self.loc9_8.2: Self]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@impl.%self.loc14_8.2: i32]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated <function> in Simple> = name_ref F, @Simple.%.loc8 [template = constants.%.3]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%F]
// CHECK:STDOUT:   %.loc18: init () = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <associated <function> in Simple> = name_ref G, @Simple.%.loc9_21 [template = constants.%.4]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element1 [template = @impl.%G]
// CHECK:STDOUT:   %.loc22_4: <bound method> = bound_method %n.ref, %.1
// CHECK:STDOUT:   %.loc22_15: init () = call %.loc22_4(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p: i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated <function> in Simple> = name_ref F, @Simple.%.loc8 [template = constants.%.3]
// CHECK:STDOUT:   %.loc26_4: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%F]
// CHECK:STDOUT:   %.loc26_16: init () = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p: i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <associated <function> in Simple> = name_ref G, @Simple.%.loc9_21 [template = constants.%.4]
// CHECK:STDOUT:   %.loc30_4.1: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element1 [template = @impl.%G]
// CHECK:STDOUT:   %.loc30_4.2: <bound method> = bound_method %.loc30_4.1, %.1
// CHECK:STDOUT:   %.loc30_4.3: i32 = bind_value %.loc30_4.1
// CHECK:STDOUT:   %.loc30_16: init () = call %.loc30_4.2(%.loc30_4.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
