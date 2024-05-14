// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn G[self: Self]();
}

impl i32 as Simple {
  // CHECK:STDERR: fail_call_invalid.carbon:[[@LINE+3]]:14: ERROR: Name `Undeclared` not found.
  // CHECK:STDERR:   fn G[self: Undeclared]();
  // CHECK:STDERR:              ^~~~~~~~~~
  fn G[self: Undeclared]();
}

fn InstanceCall(n: i32) {
  n.(Simple.G)();
}

// CHECK:STDOUT: --- fail_call_invalid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Simple [template]
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %G.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: G = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Simple, G [template]
// CHECK:STDOUT:   %.4: <associated G in Simple> = assoc_entity element0, @Simple.%G.decl [template]
// CHECK:STDOUT:   %G.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %InstanceCall: type = fn_type @InstanceCall [template]
// CHECK:STDOUT:   %struct.3: InstanceCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, %Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall.decl: InstanceCall = fn_decl @InstanceCall [template = constants.%struct.3] {
// CHECK:STDOUT:     %n.loc18_17.1: i32 = param n
// CHECK:STDOUT:     @InstanceCall.%n: i32 = bind_name n, %n.loc18_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: Simple = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: Simple = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_14.1: type = facet_type_access %Self.ref [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %Self.ref, %.loc8_14.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc8_8.1: Self = param self
// CHECK:STDOUT:     %self.loc8_8.2: Self = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_21: <associated G in Simple> = assoc_entity element0, %G.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %.loc8_21
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as Simple {
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Undeclared.ref: <error> = name_ref Undeclared, <error> [template = <error>]
// CHECK:STDOUT:     %self.loc15_8.1: <error> = param self
// CHECK:STDOUT:     %self.loc15_8.2: <error> = bind_name self, %self.loc15_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[@Simple.%self.loc8_8.2: Self]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@impl.%self.loc15_8.2: <error>]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <associated G in Simple> = name_ref G, @Simple.%.loc8_21 [template = constants.%.4]
// CHECK:STDOUT:   %.1: G = interface_witness_access @impl.%.1, element0 [template = <error>]
// CHECK:STDOUT:   %.loc19: <bound method> = bound_method %n.ref, %.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
