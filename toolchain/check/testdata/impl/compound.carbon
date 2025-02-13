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
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %Self.3c9: %Simple.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.e2e: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.df8: %F.type.e2e = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type %Simple.type [concrete]
// CHECK:STDOUT:   %assoc0.9ca: %Simple.assoc_type = assoc_entity element0, @Simple.%F.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.716: type = facet_access_type %Self.3c9 [symbolic]
// CHECK:STDOUT:   %G.type.b60: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.cb0: %G.type.b60 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Simple.assoc_type = assoc_entity element1, @Simple.%G.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %impl_witness.5b2: <witness> = impl_witness (@impl.1.%F.decl, @impl.1.%G.decl) [concrete]
// CHECK:STDOUT:   %F.type.758: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.df1: %F.type.758 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.c98: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.e73: %G.type.c98 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %i32, %impl_witness.5b2 [concrete]
// CHECK:STDOUT:   %NonInstanceCall.type: type = fn_type @NonInstanceCall [concrete]
// CHECK:STDOUT:   %NonInstanceCall: %NonInstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %F.type.e2e, %Simple.facet [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b73: type = fn_type_with_self_type %G.type.b60, %Simple.facet [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect.type: type = fn_type @NonInstanceCallIndirect [concrete]
// CHECK:STDOUT:   %NonInstanceCallIndirect: %NonInstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect.type: type = fn_type @InstanceCallIndirect [concrete]
// CHECK:STDOUT:   %InstanceCallIndirect: %InstanceCallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .NonInstanceCall = %NonInstanceCall.decl
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:     .NonInstanceCallIndirect = %NonInstanceCallIndirect.decl
// CHECK:STDOUT:     .InstanceCallIndirect = %InstanceCallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [concrete = constants.%Simple.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.1.%F.decl, @impl.1.%G.decl) [concrete = constants.%impl_witness.5b2]
// CHECK:STDOUT:   %NonInstanceCall.decl: %NonInstanceCall.type = fn_decl @NonInstanceCall [concrete = constants.%NonInstanceCall] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonInstanceCallIndirect.decl: %NonInstanceCallIndirect.type = fn_decl @NonInstanceCallIndirect [concrete = constants.%NonInstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCallIndirect.decl: %InstanceCallIndirect.type = fn_decl @InstanceCallIndirect [concrete = constants.%InstanceCallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc33: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3c9]
// CHECK:STDOUT:   %F.decl: %F.type.e2e = fn_decl @F.1 [concrete = constants.%F.df8] {} {}
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.9ca]
// CHECK:STDOUT:   %G.decl: %G.type.b60 = fn_decl @G.1 [concrete = constants.%G.cb0] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc13_14.1 (%Self.as_type.716) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc13_14.1 (%Self.as_type.716) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc13_14.1 (%Self.as_type.716) = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_14.1: type = splice_block %.loc13_14.2 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.716)] {
// CHECK:STDOUT:       %Self.ref: %Simple.type = name_ref Self, @Simple.%Self [symbolic = %Self (constants.%Self.3c9)]
// CHECK:STDOUT:       %Self.as_type.loc13_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.716)]
// CHECK:STDOUT:       %.loc13_14.2: type = converted %Self.ref, %Self.as_type.loc13_14.2 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.716)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc13_14.1 (%Self.as_type.716) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Simple.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %i32 as %Simple.ref {
// CHECK:STDOUT:   %F.decl: %F.type.758 = fn_decl @F.2 [concrete = constants.%F.df1] {} {}
// CHECK:STDOUT:   %G.decl: %G.type.c98 = fn_decl @G.2 [concrete = constants.%G.e73] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3c9)]
// CHECK:STDOUT:   %Self.as_type.loc13_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.716)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc13_14.1 (%Self.as_type.716)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %i32]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCall(%n.param_patt: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %F.ref: %Simple.assoc_type = name_ref F, @Simple.%assoc0 [concrete = constants.%assoc0.9ca]
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%impl_witness.5b2, element0 [concrete = constants.%F.df1]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param_patt: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %G.ref: %Simple.assoc_type = name_ref G, @Simple.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.b73 = impl_witness_access constants.%impl_witness.5b2, element1 [concrete = constants.%G.e73]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonInstanceCallIndirect(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %F.ref: %Simple.assoc_type = name_ref F, @Simple.%assoc0 [concrete = constants.%assoc0.9ca]
// CHECK:STDOUT:   %.loc30: ref %i32 = deref %p.ref
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%impl_witness.5b2, element0 [concrete = constants.%F.df1]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallIndirect(%p.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %G.ref: %Simple.assoc_type = name_ref G, @Simple.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %.loc34_4.1: ref %i32 = deref %p.ref
// CHECK:STDOUT:   %impl.elem1: %.b73 = impl_witness_access constants.%impl_witness.5b2, element1 [concrete = constants.%G.e73]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc34_4.1, %impl.elem1
// CHECK:STDOUT:   %.loc34_4.2: %i32 = bind_value %.loc34_4.1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%.loc34_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.3c9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.3c9) {
// CHECK:STDOUT:   %Self => constants.%Self.3c9
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => constants.%Self.as_type.716
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Simple.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Simple.facet) {
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
