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
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.c62: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.G.type.682: type = fn_type @Simple.WithSelf.G, @Simple.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Simple.WithSelf.G.d0a: %Simple.WithSelf.G.type.682 = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type @Simple [concrete]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.WithSelf.%Simple.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @i32.as.Simple.impl.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.type.321a2c.1: type = fn_type @i32.as.Simple.impl.G.loc24_27.1 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.d407b2.1: %i32.as.Simple.impl.G.type.321a2c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %i32, (%Simple.impl_witness) [concrete]
// CHECK:STDOUT:   %Simple.WithSelf.G.type.780: type = fn_type @Simple.WithSelf.G, @Simple.WithSelf(%Simple.facet) [concrete]
// CHECK:STDOUT:   %Simple.WithSelf.G.dbb: %Simple.WithSelf.G.type.780 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.type.321a2c.2: type = fn_type @i32.as.Simple.impl.G.loc24_27.2 [concrete]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.d407b2.2: %i32.as.Simple.impl.G.type.321a2c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %InstanceCall.type: type = fn_type @InstanceCall [concrete]
// CHECK:STDOUT:   %InstanceCall: %InstanceCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %.785: type = fn_type_with_self_type %Simple.WithSelf.G.type.780, %Simple.facet [concrete]
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
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InstanceCall.decl: %InstanceCall.type = fn_decl @InstanceCall [concrete = constants.%InstanceCall] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Simple.WithSelf.decl = interface_with_self_decl @Simple [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Simple.WithSelf.G.decl: @Simple.WithSelf.%Simple.WithSelf.G.type (%Simple.WithSelf.G.type.682) = fn_decl @Simple.WithSelf.G [symbolic = @Simple.WithSelf.%Simple.WithSelf.G (constants.%Simple.WithSelf.G.d0a)] {
// CHECK:STDOUT:     %self.param_patt: @Simple.WithSelf.G.%pattern_type (%pattern_type.c62) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Simple.WithSelf.G.%pattern_type (%pattern_type.c62) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Simple.WithSelf.G.%Self.as_type.loc16_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Simple.type = name_ref Self, @Simple.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc16_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref, %Self.as_type.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Simple.WithSelf.G.%Self.as_type.loc16_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %Simple.WithSelf.G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @Simple.WithSelf.%assoc0
// CHECK:STDOUT:   .Undeclared = <poisoned>
// CHECK:STDOUT:   witness = (@Simple.WithSelf.%Simple.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.Simple.impl: %i32 as %Simple.ref {
// CHECK:STDOUT:   %i32.as.Simple.impl.G.decl.loc24_27.1: %i32.as.Simple.impl.G.type.321a2c.1 = fn_decl @i32.as.Simple.impl.G.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.d407b2.1] {
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: <error> = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Undeclared.ref: <error> = name_ref Undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = value_binding self, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32.as.Simple.impl.G.decl.loc24_27.2: %i32.as.Simple.impl.G.type.321a2c.2 = fn_decl @i32.as.Simple.impl.G.loc24_27.2 [concrete = constants.%i32.as.Simple.impl.G.d407b2.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %self: %i32 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (%i32.as.Simple.impl.G.decl.loc24_27.2), @i32.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Undeclared = <poisoned>
// CHECK:STDOUT:   .G = %i32.as.Simple.impl.G.decl.loc24_27.1
// CHECK:STDOUT:   extend %Simple.ref
// CHECK:STDOUT:   witness = %Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple.WithSelf.G(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc16_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_14.1 [symbolic = %pattern_type (constants.%pattern_type.c62)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Simple.WithSelf.G.%Self.as_type.loc16_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.Simple.impl.G.loc24_27.1(%self.param: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.Simple.impl.G.loc24_27.2(%self.param: %i32) [thunk @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1 for @Simple.WithSelf.%Simple.WithSelf.G.decl, @Simple.WithSelf.G(constants.%Simple.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %i32.as.Simple.impl.G.type.321a2c.1 = name_ref G, @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.d407b2.1]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.bound: <bound method> = bound_method %self.param, %G.ref
// CHECK:STDOUT:   %i32.as.Simple.impl.G.call: init %empty_tuple.type = call %i32.as.Simple.impl.G.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCall(%n.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   %G.ref.loc28_12: %Simple.assoc_type = name_ref G, @Simple.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.785 = impl_witness_access constants.%Simple.impl_witness, element0 [concrete = constants.%i32.as.Simple.impl.G.d407b2.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %G.ref.loc28_16: %i32.as.Simple.impl.G.type.321a2c.1 = name_ref G, @i32.as.Simple.impl.%i32.as.Simple.impl.G.decl.loc24_27.1 [concrete = constants.%i32.as.Simple.impl.G.d407b2.1]
// CHECK:STDOUT:   %i32.as.Simple.impl.G.bound: <bound method> = bound_method %n.ref, %G.ref.loc28_16
// CHECK:STDOUT:   %i32.as.Simple.impl.G.call: init %empty_tuple.type = call %i32.as.Simple.impl.G.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Simple.WithSelf.G.type => constants.%Simple.WithSelf.G.type.682
// CHECK:STDOUT:   %Simple.WithSelf.G => constants.%Simple.WithSelf.G.d0a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Simple.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Simple.WithSelf.G.type => constants.%Simple.WithSelf.G.type.780
// CHECK:STDOUT:   %Simple.WithSelf.G => constants.%Simple.WithSelf.G.dbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.G(constants.%Simple.facet) {
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
