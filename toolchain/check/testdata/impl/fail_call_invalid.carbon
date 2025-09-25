// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_call_invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_call_invalid.carbon

interface Simple {
  fn G[self: Self]();
}

impl i32 as Simple {
  // CHECK:STDERR: fail_call_invalid.carbon:[[@LINE+4]]:14: error: name `Undeclared` not found [NameNotFound]
  // CHECK:STDERR:   fn G[self: Undeclared]();
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR:
  fn G[self: Undeclared]();
}

fn InstanceCall(n: i32) {
  n.(Simple.G)();
}

// CHECK:STDOUT: --- fail_call_invalid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.42f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Simple.G.type: type = fn_type @Simple.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Simple.G: %Simple.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type @Simple [concrete]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.%Simple.G.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness file.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.type.4745bc.1: type = fn_type @i32.as.Simple.impl.G.loc24_27.1 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.2bbe53.1: %i32.as.Simple.impl.G.type.4745bc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %i32, (%Simple.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.type.4745bc.2: type = fn_type @i32.as.Simple.impl.G.loc24_27.2 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.2bbe53.2: %i32.as.Simple.impl.G.type.4745bc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.cb3: type = fn_type_with_self_type %Simple.G.type, %Simple.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .Undeclared = <poisoned>
// CHECK:STDOUT:     .InstanceCall = %InstanceCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [concrete = constants.%Simple.type] {} {}
// CHECK:STDOUT:   impl_decl @i32.as.Simple.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (@i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.2), @i32.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Simple.G.decl: %Simple.G.type = fn_decl @Simple.G [concrete = constants.%Simple.G] {
// CHECK:STDOUT:     %self.patt: @Simple.G.%pattern_type (%pattern_type.42f) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Simple.G.%pattern_type (%pattern_type.42f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Simple.G.%Self.as_type.loc16_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Simple.type = name_ref Self, @Simple.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc16_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref, %Self.as_type.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Simple.G.%Self.as_type.loc16_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %Simple.G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%Simple.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.Simple.impl: %i32 as %Simple.ref {
// CHECK:STDOUT:   %i32.as.Simple.impl.G.decl.loc24_27.1: %i32.as.Simple.impl.G.type.4745bc.1 = fn_decl @i32.as.Simple.impl.G.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.2bbe53.1] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Undeclared.ref: <error> = name_ref Undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.as.Simple.impl.G.decl.loc24_27.2: %i32.as.Simple.impl.G.type.4745bc.2 = fn_decl @i32.as.Simple.impl.G.loc24_27.2 [concrete = constants.%i32.as.Simple.impl.G.2bbe53.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7ce = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ce = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Undeclared = <poisoned>
// CHECK:STDOUT:   .G = %i32.as.Simple.impl.G.decl.loc24_27.1
// CHECK:STDOUT:   witness = file.%Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple.G(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc16_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_14.1 [symbolic = %pattern_type (constants.%pattern_type.42f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Simple.G.%Self.as_type.loc16_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.Simple.impl.G.loc24_27.1(%self.param: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.Simple.impl.G.loc24_27.2(%self.param: %i32) [thunk @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %i32.as.Simple.impl.G.type.4745bc.1 = name_ref G, @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.2bbe53.1]
// CHECK:STDOUT:   %self.ref: %i32 = name_ref self, %self.param
// CHECK:STDOUT:   %i32.as.Simple.impl.G.bound: <bound method> = bound_method %self.ref, %G.ref
// CHECK:STDOUT:   %i32.as.Simple.impl.G.call: init %empty_tuple.type = call %i32.as.Simple.impl.G.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %G.ref.loc28_12: %Simple.assoc_type = name_ref G, @Simple.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.cb3 = impl_witness_access constants.%Simple.impl_witness, element0 [concrete = constants.%i32.as.Simple.impl.G.2bbe53.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %G.ref.loc28_16: %i32.as.Simple.impl.G.type.4745bc.1 = name_ref G, @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.2bbe53.1]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.bound: <bound method> = bound_method %n.ref, %G.ref.loc28_16
// CHECK:STDOUT:   %i32.as.Simple.impl.G.call: init %empty_tuple.type = call %i32.as.Simple.impl.G.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.42f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.G(constants.%Simple.facet) {
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
