// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/extend_require_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/extend_require_call.carbon

interface I {}
interface J {}

fn And(a: type, b: type) -> type = "type.and";

//@dump-sem-ir-begin
constraint C {
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
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.d87: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.d87 [symbolic]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = constraint_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d87]
// CHECK:STDOUT:   %C.WithSelf.decl = constraint_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %C.WithSelf.Self.as_type.impls.facet_type.require.decl = require_decl @C.WithSelf.Self.as_type.impls.facet_type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc21_18.1 impls %And.call
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc21_18.1: type = facet_access_type @C.%Self [symbolic = %Self.as_type.loc21_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %And.ref: %And.type = name_ref And, file.%And.decl [concrete = constants.%And]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %And.call: init type = call %And.ref(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: init type = specific_constant @C.WithSelf.Self.as_type.impls.facet_type.require.%And.call, @C.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.d87) [concrete = constants.%facet_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .And = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .And = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   extend @C.WithSelf.%.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @C.WithSelf.Self.as_type.impls.facet_type.require {
// CHECK:STDOUT:     require @C.WithSelf.Self.as_type.impls.facet_type.require.%Self.as_type.loc21_18.1 impls @C.WithSelf.Self.as_type.impls.facet_type.require.%And.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @C.WithSelf.Self.as_type.impls.facet_type.require(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d87)]
// CHECK:STDOUT:   %Self.as_type.loc21_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc21_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.d87) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.d87) {
// CHECK:STDOUT:   %Self => constants.%Self.d87
// CHECK:STDOUT:   %Self.as_type.loc21_18.2 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
