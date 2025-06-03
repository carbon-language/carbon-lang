// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/dot_self_symbolic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/dot_self_symbolic_type.carbon

// --- fail_dot_self_symbolic_type.carbon

library "[[@TEST_NAME]]";

interface A(AA:! type) { let X:! type; }

class C(CC:! type) {
  //@dump-sem-ir-begin
  fn F(T:! A(CC) where .X = CC) {}
  //@dump-sem-ir-end
}

class D(DD:! type) {
  //@dump-sem-ir-begin
  fn G(T:! type) {
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE+7]]:5: error: cannot convert type `T` into type implementing `A(DD) where .(A(DD).X) = DD` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE-10]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A(CC) where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
  }
  //@dump-sem-ir-end
}

fn H(T:! type) {
  D({}).G(T);
}

// CHECK:STDOUT: --- fail_dot_self_symbolic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e1b5.1: type = class_type @C, @C(%CC) [symbolic]
// CHECK:STDOUT:   %A.type.75a876.2: type = facet_type <@A, @A(%CC)> [symbolic]
// CHECK:STDOUT:   %.Self.fafe34.1: %A.type.75a876.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.2: type = assoc_entity_type @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.2: %A.assoc_type.ed3b32.2 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %require_complete.6daa34.1: <witness> = require_complete_type %A.type.75a876.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1bbfba.1: type = facet_access_type %.Self.fafe34.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.4e49fb.1: <witness> = lookup_impl_witness %.Self.fafe34.1, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %impl.elem0.452c56.1: type = impl_witness_access %A.lookup_impl_witness.4e49fb.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.5b404c.1: type = facet_type <@A, @A(%CC) where %impl.elem0.452c56.1 = %CC> [symbolic]
// CHECK:STDOUT:   %T.dad: %A_where.type.5b404c.1 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.2e945c.1: type = pattern_type %A_where.type.5b404c.1 [symbolic]
// CHECK:STDOUT:   %F.type.704ae6.1: type = fn_type @F, @C(%CC) [symbolic]
// CHECK:STDOUT:   %F.2d247e.1: %F.type.704ae6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic]
// CHECK:STDOUT:   %D.432: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %G.type.a94: type = fn_type @G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %G.714: %G.type.a94 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.f2e1b5.2: type = class_type @C, @C(%DD) [symbolic]
// CHECK:STDOUT:   %F.type.704ae6.2: type = fn_type @F, @C(%DD) [symbolic]
// CHECK:STDOUT:   %F.2d247e.2: %F.type.704ae6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e1b5.2 [symbolic]
// CHECK:STDOUT:   %A.type.75a876.3: type = facet_type <@A, @A(%DD)> [symbolic]
// CHECK:STDOUT:   %.Self.fafe34.2: %A.type.75a876.3 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.4e49fb.2: <witness> = lookup_impl_witness %.Self.fafe34.2, @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %impl.elem0.452c56.2: type = impl_witness_access %A.lookup_impl_witness.4e49fb.2, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.5b404c.2: type = facet_type <@A, @A(%DD) where %impl.elem0.452c56.2 = %DD> [symbolic]
// CHECK:STDOUT:   %require_complete.6daa34.2: <witness> = require_complete_type %A.type.75a876.3 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.3: type = assoc_entity_type @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.3: %A.assoc_type.ed3b32.3 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1bbfba.2: type = facet_access_type %.Self.fafe34.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2e945c.2: type = pattern_type %A_where.type.5b404c.2 [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %G.type.0da: type = fn_type @G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %G.54c: %G.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.type.789: type = fn_type @F, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.cc7: %F.type.789 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.700: type = facet_type <@A, @A(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.Self.4e2: %A.type.700 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.935: <witness> = lookup_impl_witness %.Self.4e2, @A, @A(%empty_struct_type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab3: type = impl_witness_access %A.lookup_impl_witness.935, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%CC.loc6_9.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%CC.loc6_9.2) [symbolic = %F.type (constants.%F.type.704ae6.1)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type.704ae6.1) = struct_value () [symbolic = %F (constants.%F.2d247e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type.704ae6.1) = fn_decl @F [symbolic = @C.%F (constants.%F.2d247e.1)] {
// CHECK:STDOUT:       %T.patt: @F.%pattern_type (%pattern_type.2e945c.1) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_18.1: type = splice_block %.loc8_18.2 [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)] {
// CHECK:STDOUT:         %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:         %CC.ref.loc8_14: type = name_ref CC, @C.%CC.loc6_9.1 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %A.type.loc8_16.2: type = facet_type <@A, @A(constants.%CC)> [symbolic = %A.type.loc8_16.1 (constants.%A.type.75a876.2)]
// CHECK:STDOUT:         <elided>
// CHECK:STDOUT:         %.Self.ref: @F.%A.type.loc8_16.1 (%A.type.75a876.2) = name_ref .Self, %.Self.2 [symbolic = %.Self.1 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:         %.loc8_24.1: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = specific_constant @X.%assoc0, @A(constants.%CC) [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %X.ref: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = name_ref X, %.loc8_24.1 [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %.Self.as_type.loc8_24.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc8_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:         %.loc8_24.2: type = converted %.Self.ref, %.Self.as_type.loc8_24.2 [symbolic = %.Self.as_type.loc8_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:         %impl.elem0.loc8_24.2: type = impl_witness_access constants.%A.lookup_impl_witness.4e49fb.1, element0 [symbolic = %impl.elem0.loc8_24.1 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:         %CC.ref.loc8_29: type = name_ref CC, @C.%CC.loc6_9.1 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %.loc8_18.2: type = where_expr %.Self.2 [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)] {
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc8_24.2, %CC.ref.loc8_29
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc8_8.2: @F.%A_where.type (%A_where.type.5b404c.1) = bind_symbolic_name T, 1 [symbolic = %T.loc8_8.1 (constants.%T.dad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e1b5.1
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .CC = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc12_9.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @D(%DD.loc12_9.2) [symbolic = %G.type (constants.%G.type.a94)]
// CHECK:STDOUT:   %G: @D.%G.type (%G.type.a94) = struct_value () [symbolic = %G (constants.%G.714)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %G.decl: @D.%G.type (%G.type.a94) = fn_decl @G [symbolic = @D.%G (constants.%G.714)] {
// CHECK:STDOUT:       %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.loc14_8.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.432
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%CC.loc6_9.1: type, %T.loc8_8.2: @F.%A_where.type (%A_where.type.5b404c.1)) {
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:   %A.type.loc8_16.1: type = facet_type <@A, @A(%CC)> [symbolic = %A.type.loc8_16.1 (constants.%A.type.75a876.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc8_16.1 [symbolic = %require_complete (constants.%require_complete.6daa34.1)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%CC) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.2)]
// CHECK:STDOUT:   %assoc0: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = assoc_entity element0, @A.%X [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:   %.Self.as_type.loc8_24.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc8_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @A, @A(%CC) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.4e49fb.1)]
// CHECK:STDOUT:   %impl.elem0.loc8_24.1: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_24.1 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%CC) where %impl.elem0.loc8_24.1 = %CC> [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)]
// CHECK:STDOUT:   %T.loc8_8.1: @F.%A_where.type (%A_where.type.5b404c.1) = bind_symbolic_name T, 1 [symbolic = %T.loc8_8.1 (constants.%T.dad)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.2e945c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@D.%DD.loc12_9.1: type, %T.loc14_8.2: type) {
// CHECK:STDOUT:   %T.loc14_8.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %C.loc22_9.2: type = class_type @C, @C(%DD) [symbolic = %C.loc22_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc22_9.2 [symbolic = %require_complete (constants.%require_complete.389)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%DD) [symbolic = %F.type (constants.%F.type.704ae6.2)]
// CHECK:STDOUT:   %F: @G.%F.type (%F.type.704ae6.2) = struct_value () [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc12_9.1 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %C.loc22_9.1: type = class_type @C, @C(constants.%DD) [symbolic = %C.loc22_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:     %.loc22: @G.%F.type (%F.type.704ae6.2) = specific_constant @C.%F.decl, @C(constants.%DD) [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:     %F.ref: @G.%F.type (%F.type.704ae6.2) = name_ref F, %.loc22 [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_8.2 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:     %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%DD, <error>) [concrete = <error>]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn() [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%CC) {
// CHECK:STDOUT:   %CC.loc6_9.2 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.704ae6.1
// CHECK:STDOUT:   %F => constants.%F.2d247e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%CC, constants.%T.dad) {
// CHECK:STDOUT:   %CC => constants.%CC
// CHECK:STDOUT:   %A.type.loc8_16.1 => constants.%A.type.75a876.2
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.fafe34.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6daa34.1
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.2
// CHECK:STDOUT:   %.Self.as_type.loc8_24.1 => constants.%.Self.as_type.1bbfba.1
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.1
// CHECK:STDOUT:   %impl.elem0.loc8_24.1 => constants.%impl.elem0.452c56.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.5b404c.1
// CHECK:STDOUT:   %T.loc8_8.1 => constants.%T.dad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2e945c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc12_9.2 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.a94
// CHECK:STDOUT:   %G => constants.%G.714
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%DD, constants.%T.336) {
// CHECK:STDOUT:   %T.loc14_8.1 => constants.%T.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%DD) {
// CHECK:STDOUT:   %CC.loc6_9.2 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.704ae6.2
// CHECK:STDOUT:   %F => constants.%F.2d247e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%DD, <error>) {
// CHECK:STDOUT:   %CC => constants.%DD
// CHECK:STDOUT:   %A.type.loc8_16.1 => constants.%A.type.75a876.3
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.fafe34.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6daa34.2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.3
// CHECK:STDOUT:   %.Self.as_type.loc8_24.1 => constants.%.Self.as_type.1bbfba.2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.2
// CHECK:STDOUT:   %impl.elem0.loc8_24.1 => constants.%impl.elem0.452c56.2
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.5b404c.2
// CHECK:STDOUT:   %T.loc8_8.1 => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2e945c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc12_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.0da
// CHECK:STDOUT:   %G => constants.%G.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_struct_type, constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc14_8.1 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc22_9.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %F.type => constants.%F.type.789
// CHECK:STDOUT:   %F => constants.%F.cc7
// CHECK:STDOUT:   %A.type => constants.%A.type.700
// CHECK:STDOUT:   %.Self => constants.%.Self.4e2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.935
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.ab3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %CC.loc6_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.789
// CHECK:STDOUT:   %F => constants.%F.cc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
