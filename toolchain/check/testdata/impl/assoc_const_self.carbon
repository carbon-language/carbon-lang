// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/assoc_const_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/assoc_const_self.carbon

interface I {
  let V:! Self;
}

impl {} as I where .V = {} {}

// CHECK:STDOUT: --- assoc_const_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, %Self.as_type [symbolic]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, <unexpected>.inst26.loc12_15 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %I.facet.1f7: %I.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %.Self.as_type [symbolic]
// CHECK:STDOUT:   %impl.elem0: %Self.as_type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct) [template]
// CHECK:STDOUT:   %I.facet.27e: %I.type = facet_value %empty_struct_type, %impl_witness [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %assoc_type = name_ref V, <unexpected>.inst31 [symbolic = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %Self.as_type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc15_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_26.2: %empty_struct_type = converted %.loc15_26.1, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc15_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = <unexpected>.inst31
// CHECK:STDOUT:   witness = (<unexpected>.inst26.loc12_15)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%Self.as_type (%Self.as_type) [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.1f7) {
// CHECK:STDOUT:   %Self => constants.%I.facet.1f7
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.27e) {
// CHECK:STDOUT:   %Self => constants.%I.facet.27e
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
