// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/extend_require_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/extend_require_call.carbon

interface I {}
interface J {}

fn And(a: type, b: type) -> type = "type.and";

//@dump-sem-ir-begin
interface K {
  // The constraint is a call, which is an initializing expression.
  extend require impls And(I, J);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- extend_require_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.a32: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.a32 [symbolic]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a32]
// CHECK:STDOUT:   %K.WithSelf.decl = interface_with_self_decl @K [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %K.WithSelf.Self.as_type.impls.facet_type.require.decl = require_decl @K.WithSelf.Self.as_type.impls.facet_type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc21_18.1 impls %And.call
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc21_18.1: type = facet_access_type @K.%Self [symbolic = %Self.as_type.loc21_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %And.ref: %And.type = name_ref And, file.%And.decl [concrete = constants.%And]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %And.call: init type = call %And.ref(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: init type = specific_constant @K.WithSelf.Self.as_type.impls.facet_type.require.%And.call, @K.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.a32) [concrete = constants.%facet_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .And = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .And = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   extend @K.WithSelf.%.loc21
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @K.WithSelf.Self.as_type.impls.facet_type.require {
// CHECK:STDOUT:     require @K.WithSelf.Self.as_type.impls.facet_type.require.%Self.as_type.loc21_18.1 impls @K.WithSelf.Self.as_type.impls.facet_type.require.%And.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @K.WithSelf.Self.as_type.impls.facet_type.require(@K.%Self: %K.type) {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.a32)]
// CHECK:STDOUT:   %Self.as_type.loc21_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc21_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf(constants.%Self.a32) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.a32) {
// CHECK:STDOUT:   %Self => constants.%Self.a32
// CHECK:STDOUT:   %Self.as_type.loc21_18.2 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
