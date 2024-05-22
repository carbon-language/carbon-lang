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
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Simple, F [template]
// CHECK:STDOUT:   %.4: <associated F in Simple> = assoc_entity element0, @Simple.%F.decl [template]
// CHECK:STDOUT:   %G.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Simple, G [template]
// CHECK:STDOUT:   %.6: <associated G in Simple> = assoc_entity element1, @Simple.%G.decl [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.3: Int32 = struct_value () [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.4: F = struct_value () [template]
// CHECK:STDOUT:   %G.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %struct.5: G = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%struct.4, %struct.5) [template]
// CHECK:STDOUT:   %NonInstanceCall: type = fn_type @NonInstanceCall [template]
// CHECK:STDOUT:   %struct.6: NonInstanceCall = struct_value () [template]
// CHECK:STDOUT:   %InstanceCall: type = fn_type @InstanceCall [template]
// CHECK:STDOUT:   %struct.7: InstanceCall = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type i32 [template]
// CHECK:STDOUT:   %NonInstanceCallIndirect: type = fn_type @NonInstanceCallIndirect [template]
// CHECK:STDOUT:   %struct.8: NonInstanceCallIndirect = struct_value () [template]
// CHECK:STDOUT:   %InstanceCallIndirect: type = fn_type @InstanceCallIndirect [template]
// CHECK:STDOUT:   %struct.9: InstanceCallIndirect = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .NonInstanceCall = %NonInstanceCall.decl
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect.decl
// CHECK:STDOUT:     .InstanceCallIndirect = %InstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc12_6.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_6.2: type = converted %int.make_type_32.loc12, %.loc12_6.1 [template = i32]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, %Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %NonInstanceCall.decl: NonInstanceCall = fn_decl @NonInstanceCall [template = constants.%struct.6] {
// CHECK:STDOUT:     %int.make_type_32.loc17: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc17_23.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:     %.loc17_23.2: type = converted %int.make_type_32.loc17, %.loc17_23.1 [template = i32]
// CHECK:STDOUT:     %n.loc17_20.1: i32 = param n
// CHECK:STDOUT:     @NonInstanceCall.%n: i32 = bind_name n, %n.loc17_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %InstanceCall.decl: InstanceCall = fn_decl @InstanceCall [template = constants.%struct.7] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc21_20.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_20.2: type = converted %int.make_type_32.loc21, %.loc21_20.1 [template = i32]
// CHECK:STDOUT:     %n.loc21_17.1: i32 = param n
// CHECK:STDOUT:     @InstanceCall.%n: i32 = bind_name n, %n.loc21_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: NonInstanceCallIndirect = fn_decl @NonInstanceCallIndirect [template = constants.%struct.8] {
// CHECK:STDOUT:     %int.make_type_32.loc25: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc25_34.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:     %.loc25_34.2: type = converted %int.make_type_32.loc25, %.loc25_34.1 [template = i32]
// CHECK:STDOUT:     %.loc25_34.3: type = ptr_type i32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc25_28.1: i32* = param p
// CHECK:STDOUT:     @NonInstanceCallIndirect.%p: i32* = bind_name p, %p.loc25_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %InstanceCallIndirect.decl: InstanceCallIndirect = fn_decl @InstanceCallIndirect [template = constants.%struct.9] {
// CHECK:STDOUT:     %int.make_type_32.loc29: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc29_31.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:     %.loc29_31.2: type = converted %int.make_type_32.loc29, %.loc29_31.1 [template = i32]
// CHECK:STDOUT:     %.loc29_31.3: type = ptr_type i32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc29_25.1: i32* = param p
// CHECK:STDOUT:     @InstanceCallIndirect.%p: i32* = bind_name p, %p.loc29_25.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc8: <associated F in Simple> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref: Simple = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_14.1: type = facet_type_access %Self.ref [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_14.2: type = converted %Self.ref, %.loc9_14.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc9_8.1: Self = param self
// CHECK:STDOUT:     %self.loc9_8.2: Self = bind_name self, %self.loc9_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_21: <associated G in Simple> = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   .G = %.loc9_21
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as Simple {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.4] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.2 [template = constants.%struct.5] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:     %.loc14_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc14_14.2: type = converted %int.make_type_32, %.loc14_14.1 [template = i32]
// CHECK:STDOUT:     %self.loc14_8.1: i32 = param self
// CHECK:STDOUT:     %self.loc14_8.2: i32 = bind_name self, %self.loc14_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl, %G.decl) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[@Simple.%self.loc9_8.2: Self]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@impl.%self.loc14_8.2: i32]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated F in Simple> = name_ref F, @Simple.%.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.1: F = interface_witness_access @impl.%.1, element0 [template = constants.%struct.4]
// CHECK:STDOUT:   %F.call: init () = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <associated G in Simple> = name_ref G, @Simple.%.loc9_21 [template = constants.%.6]
// CHECK:STDOUT:   %.1: G = interface_witness_access @impl.%.1, element1 [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc22: <bound method> = bound_method %n.ref, %.1
// CHECK:STDOUT:   %G.call: init () = call %.loc22(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p: i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated F in Simple> = name_ref F, @Simple.%.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc26: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.1: F = interface_witness_access @impl.%.1, element0 [template = constants.%struct.4]
// CHECK:STDOUT:   %F.call: init () = call %.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p: i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <associated G in Simple> = name_ref G, @Simple.%.loc9_21 [template = constants.%.6]
// CHECK:STDOUT:   %.loc30_4.1: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.1: G = interface_witness_access @impl.%.1, element1 [template = constants.%struct.5]
// CHECK:STDOUT:   %.loc30_4.2: <bound method> = bound_method %.loc30_4.1, %.1
// CHECK:STDOUT:   %.loc30_4.3: i32 = bind_value %.loc30_4.1
// CHECK:STDOUT:   %G.call: init () = call %.loc30_4.2(%.loc30_4.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
