// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/compound.carbon

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
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Simple.%F.decl [template]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.1, %G.type.1 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @Simple.%G.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%F.2, %G.2) [template]
// CHECK:STDOUT:   %NonInstanceCall.type: type = fn_type @NonInstanceCall [template]
// CHECK:STDOUT:   %NonInstanceCall: %NonInstanceCall.type = struct_value () [template]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [template]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type i32 [template]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [template]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [template]
// CHECK:STDOUT:   %InstanceCallIndirect.type: type = fn_type @InstanceCallIndirect [template]
// CHECK:STDOUT:   %InstanceCallIndirect: %InstanceCallIndirect.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .NonInstanceCall = %NonInstanceCall.decl
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect.decl
// CHECK:STDOUT:     .InstanceCallIndirect = %InstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_6.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:     %.loc16_6.2: type = converted %int.make_type_32.loc16, %.loc16_6.1 [template = i32]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, %Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCall.decl: %NonInstanceCall.type = fn_decl @NonInstanceCall [template = constants.%NonInstanceCall] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_23.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_23.2: type = converted %int.make_type_32.loc21, %.loc21_23.1 [template = i32]
// CHECK:STDOUT:     %n.loc21_20.1: i32 = param n
// CHECK:STDOUT:     @NonInstanceCall.%n: i32 = bind_name n, %n.loc21_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [template = constants.%InstanceCall] {
// CHECK:STDOUT:     %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_20.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:     %.loc25_20.2: type = converted %int.make_type_32.loc25, %.loc25_20.1 [template = i32]
// CHECK:STDOUT:     %n.loc25_17.1: i32 = param n
// CHECK:STDOUT:     @InstanceCall.%n: i32 = bind_name n, %n.loc25_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [template = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_34.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:     %.loc29_34.2: type = converted %int.make_type_32.loc29, %.loc29_34.1 [template = i32]
// CHECK:STDOUT:     %.loc29_34.3: type = ptr_type i32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc29_28.1: %.8 = param p
// CHECK:STDOUT:     @NonInstanceCallIndirect.%p: %.8 = bind_name p, %p.loc29_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect.decl: %InstanceCallIndirect.type = fn_decl @InstanceCallIndirect [template = constants.%InstanceCallIndirect] {
// CHECK:STDOUT:     %int.make_type_32.loc33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc33_31.1: type = value_of_initializer %int.make_type_32.loc33 [template = i32]
// CHECK:STDOUT:     %.loc33_31.2: type = converted %int.make_type_32.loc33, %.loc33_31.1 [template = i32]
// CHECK:STDOUT:     %.loc33_31.3: type = ptr_type i32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc33_25.1: %.8 = param p
// CHECK:STDOUT:     @InstanceCallIndirect.%p: %.8 = bind_name p, %p.loc33_25.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {}
// CHECK:STDOUT:   %.loc12: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {
// CHECK:STDOUT:     %Self.ref: %.1 = name_ref Self, %Self [symbolic = @G.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc13_14.1: type = facet_type_access %Self.ref [symbolic = @G.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc13_14.2: type = converted %Self.ref, %.loc13_14.1 [symbolic = @G.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %self.loc13_8.1: @G.1.%Self (%Self) = param self
// CHECK:STDOUT:     %self.loc13_8.2: @G.1.%Self (%Self) = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_21: %.5 = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   .G = %.loc13_21
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {}
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc18_14.2: type = converted %int.make_type_32, %.loc18_14.1 [template = i32]
// CHECK:STDOUT:     %self.loc18_8.1: i32 = param self
// CHECK:STDOUT:     %self.loc18_8.2: i32 = bind_name self, %self.loc18_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = interface_witness (%F.decl, %G.decl) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Simple.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@Simple.%Self: %.1) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@Simple.%self.loc13_8.2: @G.1.%Self (%Self)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@impl.%self.loc18_8.2: i32]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: %.3 = name_ref F, @Simple.%.loc12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc22: %F.type.1 = interface_witness_access @impl.%.loc16, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call: init %.2 = call %.loc22()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: %.5 = name_ref G, @Simple.%.loc13_21 [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_4.1: %G.type.1 = interface_witness_access @impl.%.loc16, element1 [template = constants.%G.2]
// CHECK:STDOUT:   %.loc26_4.2: <bound method> = bound_method %n.ref, %.loc26_4.1
// CHECK:STDOUT:   %G.call: init %.2 = call %.loc26_4.2(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p: %.8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: %.3 = name_ref F, @Simple.%.loc12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc30_4.1: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.loc30_4.2: %F.type.1 = interface_witness_access @impl.%.loc16, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call: init %.2 = call %.loc30_4.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p: %.8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: %.5 = name_ref G, @Simple.%.loc13_21 [template = constants.%.6]
// CHECK:STDOUT:   %.loc34_4.1: ref i32 = deref %p.ref
// CHECK:STDOUT:   %.loc34_4.2: %G.type.1 = interface_witness_access @impl.%.loc16, element1 [template = constants.%G.2]
// CHECK:STDOUT:   %.loc34_4.3: <bound method> = bound_method %.loc34_4.1, %.loc34_4.2
// CHECK:STDOUT:   %.loc34_4.4: i32 = bind_value %.loc34_4.1
// CHECK:STDOUT:   %G.call: init %.2 = call %.loc34_4.3(%.loc34_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(i32) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(i32) {
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
