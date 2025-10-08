// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
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
  // The `.Self` bind_symbolic_name here should have a
  // symbolic constant value, not symbolic_self, because
  // its type depends on `CC`.
  fn F(T:! A(CC) where .X = CC) {}
  //@dump-sem-ir-end
}

class D(DD:! type) {
  fn G(T:! type) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE+7]]:5: error: cannot convert type `T` into type implementing `A(DD) where .(A(DD).X) = DD` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE-10]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A(CC) where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
    //@dump-sem-ir-end
  }
}

fn H(T:! type) {
  D({}).G(T);
}

// --- dot_self_symbolic_in_impl_lookup.carbon

library "[[@TEST_NAME]]";

interface A(AA:! type) { let X:! type; }

interface B(AA:! type) {}

//@dump-sem-ir-begin
impl forall [AA:! type, BB:! A(AA) where .X = ()] BB as B(AA*) {}
//@dump-sem-ir-end

impl forall [AA:! type, BB:! type] BB as B(AA) {}

class D(DD:! type) {
  fn G(T:! type) {
    // TODO: This produces a `lookup_impl_witness` instruction looking
    // for `.Self` as `A(DD)` in the generic eval block for `G`. That
    // is from an impl we ended up not using, so shouldn't be included
    // in the eval block.
    //@dump-sem-ir-begin
    T as B(DD*);
    //@dump-sem-ir-end
  }
}

fn H(T:! type) {
  D({}).G(T);
}

// CHECK:STDOUT: --- fail_dot_self_symbolic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e1b5.1: type = class_type @C, @C(%CC) [symbolic]
// CHECK:STDOUT:   %A.type.d88d48.2: type = facet_type <@A, @A(%CC)> [symbolic]
// CHECK:STDOUT:   %.Self.c063ed.1: %A.type.d88d48.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.2: type = assoc_entity_type @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.2: %A.assoc_type.ed3b32.2 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %require_complete.34a: <witness> = require_complete_type %A.type.d88d48.2 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c063ed.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.ddb77a.1: <witness> = lookup_impl_witness %.Self.c063ed.1, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7afc4b.1: type = impl_witness_access %A.lookup_impl_witness.ddb77a.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.6f52c8.1: type = facet_type <@A, @A(%CC) where %impl.elem0.7afc4b.1 = %CC> [symbolic]
// CHECK:STDOUT:   %T.140: %A_where.type.6f52c8.1 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ad5: type = pattern_type %A_where.type.6f52c8.1 [symbolic]
// CHECK:STDOUT:   %C.F.type.704ae6.1: type = fn_type @C.F, @C(%CC) [symbolic]
// CHECK:STDOUT:   %C.F.2d247e.1: %C.F.type.704ae6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic]
// CHECK:STDOUT:   %D.432: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.a94: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.714: %D.G.type.a94 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.f2e1b5.2: type = class_type @C, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.type.704ae6.2: type = fn_type @C.F, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.2d247e.2: %C.F.type.704ae6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e1b5.2 [symbolic]
// CHECK:STDOUT:   %A.type.d88d48.3: type = facet_type <@A, @A(%DD)> [symbolic]
// CHECK:STDOUT:   %.Self.c063ed.2: %A.type.d88d48.3 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.ddb77a.2: <witness> = lookup_impl_witness %.Self.c063ed.2, @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7afc4b.2: type = impl_witness_access %A.lookup_impl_witness.ddb77a.2, element0 [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.0da: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.54c: %D.G.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.type.789: type = fn_type @C.F, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.cc7: %C.F.type.789 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.23f: type = facet_type <@A, @A(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.Self.495: %A.type.23f = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.37c: <witness> = lookup_impl_witness %.Self.495, @A, @A(%empty_struct_type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e27: type = impl_witness_access %A.lookup_impl_witness.37c, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%CC.loc6_9.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%CC.loc6_9.1) [symbolic = %C.F.type (constants.%C.F.type.704ae6.1)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type.704ae6.1) = struct_value () [symbolic = %C.F (constants.%C.F.2d247e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type.704ae6.1) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F.2d247e.1)] {
// CHECK:STDOUT:       %T.patt: @C.F.%pattern_type (%pattern_type.ad5) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %.loc11_18.2 [symbolic = %A_where.type (constants.%A_where.type.6f52c8.1)] {
// CHECK:STDOUT:         <elided>
// CHECK:STDOUT:         %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:         %CC.ref.loc11_14: type = name_ref CC, @C.%CC.loc6_9.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %A.type.loc11_16.2: type = facet_type <@A, @A(constants.%CC)> [symbolic = %A.type.loc11_16.1 (constants.%A.type.d88d48.2)]
// CHECK:STDOUT:         <elided>
// CHECK:STDOUT:         %.Self.ref: @C.F.%A.type.loc11_16.1 (%A.type.d88d48.2) = name_ref .Self, %.Self.3 [symbolic = %.Self.1 (constants.%.Self.c063ed.1)]
// CHECK:STDOUT:         %.loc11_24.1: @C.F.%A.assoc_type (%A.assoc_type.ed3b32.2) = specific_constant @X.%assoc0, @A(constants.%CC) [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %X.ref: @C.F.%A.assoc_type (%A.assoc_type.ed3b32.2) = name_ref X, %.loc11_24.1 [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:         %.loc11_24.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:         %impl.elem0.loc11_24.2: type = impl_witness_access constants.%A.lookup_impl_witness.ddb77a.1, element0 [symbolic = %impl.elem0.loc11_24.1 (constants.%impl.elem0.7afc4b.1)]
// CHECK:STDOUT:         %CC.ref.loc11_29: type = name_ref CC, @C.%CC.loc6_9.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %.loc11_18.2: type = where_expr %.Self.3 [symbolic = %A_where.type (constants.%A_where.type.6f52c8.1)] {
// CHECK:STDOUT:           requirement_base_facet_type constants.%A.type.d88d48.2
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc11_24.2, %CC.ref.loc11_29
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc11_8.2: @C.F.%A_where.type (%A_where.type.6f52c8.1) = bind_symbolic_name T, 1 [symbolic = %T.loc11_8.1 (constants.%T.140)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e1b5.1
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .CC = <poisoned>
// CHECK:STDOUT:     .F = %C.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc15_9.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.432
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%CC.loc6_9.2: type, %T.loc11_8.2: @C.F.%A_where.type (%A_where.type.6f52c8.1)) {
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:   %A.type.loc11_16.1: type = facet_type <@A, @A(%CC)> [symbolic = %A.type.loc11_16.1 (constants.%A.type.d88d48.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc11_16.1 [symbolic = %require_complete (constants.%require_complete.34a)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%CC) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.2)]
// CHECK:STDOUT:   %assoc0: @C.F.%A.assoc_type (%A.assoc_type.ed3b32.2) = assoc_entity element0, @A.%X [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.1 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @A, @A(%CC) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.ddb77a.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_24.1: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_24.1 (constants.%impl.elem0.7afc4b.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%CC) where %impl.elem0.loc11_24.1 = %CC> [symbolic = %A_where.type (constants.%A_where.type.6f52c8.1)]
// CHECK:STDOUT:   %T.loc11_8.1: @C.F.%A_where.type (%A_where.type.6f52c8.1) = bind_symbolic_name T, 1 [symbolic = %T.loc11_8.1 (constants.%T.140)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.ad5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc15_9.2: type, %T.loc16_8.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %C.loc25_9.2: type = class_type @C, @C(%DD) [symbolic = %C.loc25_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc25_9.2 [symbolic = %require_complete (constants.%require_complete.389)]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%DD) [symbolic = %C.F.type (constants.%C.F.type.704ae6.2)]
// CHECK:STDOUT:   %C.F: @D.G.%C.F.type (%C.F.type.704ae6.2) = struct_value () [symbolic = %C.F (constants.%C.F.2d247e.2)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%DD)> [symbolic = %A.type (constants.%A.type.d88d48.3)]
// CHECK:STDOUT:   %.Self.loc11: @D.G.%A.type (%A.type.d88d48.3) = bind_symbolic_name .Self [symbolic = %.Self.loc11 (constants.%.Self.c063ed.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc11, @A, @A(%DD) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.ddb77a.2)]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.7afc4b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc15_9.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %C.loc25_9.1: type = class_type @C, @C(constants.%DD) [symbolic = %C.loc25_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:     %.loc25: @D.G.%C.F.type (%C.F.type.704ae6.2) = specific_constant @C.%C.F.decl, @C(constants.%DD) [symbolic = %C.F (constants.%C.F.2d247e.2)]
// CHECK:STDOUT:     %F.ref: @D.G.%C.F.type (%C.F.type.704ae6.2) = name_ref F, %.loc25 [symbolic = %C.F (constants.%C.F.2d247e.2)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_8.2 [symbolic = %T.loc16_8.1 (constants.%T.336)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%CC) {
// CHECK:STDOUT:   %CC.loc6_9.1 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.704ae6.1
// CHECK:STDOUT:   %C.F => constants.%C.F.2d247e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%CC, constants.%T.140) {
// CHECK:STDOUT:   %CC => constants.%CC
// CHECK:STDOUT:   %A.type.loc11_16.1 => constants.%A.type.d88d48.2
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.c063ed.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.34a
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.2
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.ddb77a.1
// CHECK:STDOUT:   %impl.elem0.loc11_24.1 => constants.%impl.elem0.7afc4b.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.6f52c8.1
// CHECK:STDOUT:   %T.loc11_8.1 => constants.%T.140
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ad5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc15_9.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.a94
// CHECK:STDOUT:   %D.G => constants.%D.G.714
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.336) {
// CHECK:STDOUT:   %T.loc16_8.1 => constants.%T.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%DD) {
// CHECK:STDOUT:   %CC.loc6_9.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.704ae6.2
// CHECK:STDOUT:   %C.F => constants.%C.F.2d247e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc15_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.0da
// CHECK:STDOUT:   %D.G => constants.%D.G.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc16_8.1 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc25_9.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.789
// CHECK:STDOUT:   %C.F => constants.%C.F.cc7
// CHECK:STDOUT:   %A.type => constants.%A.type.23f
// CHECK:STDOUT:   %.Self.loc11 => constants.%.Self.495
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.37c
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.e27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %CC.loc6_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.789
// CHECK:STDOUT:   %C.F => constants.%C.F.cc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dot_self_symbolic_in_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AA: type = bind_symbolic_name AA, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.d88d48.1: type = facet_type <@A, @A(%AA)> [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %B.type.bf2: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.bf2 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.c063ed.1: %A.type.d88d48.1 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete.34a: <witness> = require_complete_type %A.type.d88d48.1 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.c063ed.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.ddb77a.1: <witness> = lookup_impl_witness %.Self.c063ed.1, @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7afc4b.1: type = impl_witness_access %A.lookup_impl_witness.ddb77a.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.ecbf58.1: type = facet_type <@A, @A(%AA) where %impl.elem0.7afc4b.1 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %BB.afa: %A_where.type.ecbf58.1 = bind_symbolic_name BB, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.22c: type = pattern_type %A_where.type.ecbf58.1 [symbolic]
// CHECK:STDOUT:   %BB.binding.as_type: type = symbolic_binding_type BB, 1, %BB.afa [symbolic]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %AA [symbolic]
// CHECK:STDOUT:   %B.type.6d756f.1: type = facet_type <@B, @B(%ptr.79f131.1)> [symbolic]
// CHECK:STDOUT:   %require_complete.672f1b.1: <witness> = require_complete_type %B.type.6d756f.1 [symbolic]
// CHECK:STDOUT:   %B.impl_witness.825: <witness> = impl_witness file.%B.impl_witness_table.loc9, @BB.binding.as_type.as.B.impl(%AA, %BB.afa) [symbolic]
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic]
// CHECK:STDOUT:   %D.432: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.a94: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.714: %D.G.type.a94 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %DD [symbolic]
// CHECK:STDOUT:   %B.type.6d756f.2: type = facet_type <@B, @B(%ptr.79f131.2)> [symbolic]
// CHECK:STDOUT:   %A.type.d88d48.2: type = facet_type <@A, @A(%DD)> [symbolic]
// CHECK:STDOUT:   %.Self.c063ed.2: %A.type.d88d48.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.ddb77a.2: <witness> = lookup_impl_witness %.Self.c063ed.2, @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7afc4b.2: type = impl_witness_access %A.lookup_impl_witness.ddb77a.2, element0 [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.8e3: <witness> = lookup_impl_witness %T.336, @B, @B(%ptr.79f131.2) [symbolic]
// CHECK:STDOUT:   %B.facet.c0e: %B.type.6d756f.2 = facet_value %T.336, (%B.lookup_impl_witness.8e3) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.0da: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.54c: %D.G.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.type.997: type = facet_type <@B, @B(%ptr.c28)> [concrete]
// CHECK:STDOUT:   %A.type.23f: type = facet_type <@A, @A(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.Self.495: %A.type.23f = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.37c: <witness> = lookup_impl_witness %.Self.495, @A, @A(%empty_struct_type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e27: type = impl_witness_access %A.lookup_impl_witness.37c, element0 [symbolic_self]
// CHECK:STDOUT:   %B.lookup_impl_witness.c4b: <witness> = lookup_impl_witness %T.8b3, @B, @B(%ptr.c28) [symbolic]
// CHECK:STDOUT:   %B.facet.b55: %B.type.997 = facet_value %T.8b3, (%B.lookup_impl_witness.c4b) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @BB.binding.as_type.as.B.impl [concrete] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %BB.patt: @BB.binding.as_type.as.B.impl.%pattern_type (%pattern_type.22c) = symbolic_binding_pattern BB, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %BB.ref: @BB.binding.as_type.as.B.impl.%A_where.type (%A_where.type.ecbf58.1) = name_ref BB, %BB.loc9_25.1 [symbolic = %BB.loc9_25.2 (constants.%BB.afa)]
// CHECK:STDOUT:     %BB.as_type: type = facet_access_type %BB.ref [symbolic = %BB.binding.as_type (constants.%BB.binding.as_type)]
// CHECK:STDOUT:     %.loc9_51: type = converted %BB.ref, %BB.as_type [symbolic = %BB.binding.as_type (constants.%BB.binding.as_type)]
// CHECK:STDOUT:     %B.ref: %B.type.bf2 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %AA.ref.loc9_59: type = name_ref AA, %AA.loc9_14.1 [symbolic = %AA.loc9_14.2 (constants.%AA)]
// CHECK:STDOUT:     %ptr.loc9_61.1: type = ptr_type %AA.ref.loc9_59 [symbolic = %ptr.loc9_61.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %B.type.loc9_62.1: type = facet_type <@B, @B(constants.%ptr.79f131.1)> [symbolic = %B.type.loc9_62.2 (constants.%B.type.6d756f.1)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %AA.loc9_14.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc9_14.2 (constants.%AA)]
// CHECK:STDOUT:     %.loc9_36.1: type = splice_block %.loc9_36.2 [symbolic = %A_where.type (constants.%A_where.type.ecbf58.1)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %AA.ref.loc9_32: type = name_ref AA, %AA.loc9_14.1 [symbolic = %AA.loc9_14.2 (constants.%AA)]
// CHECK:STDOUT:       %A.type.loc9_34.1: type = facet_type <@A, @A(constants.%AA)> [symbolic = %A.type.loc9_34.2 (constants.%A.type.d88d48.1)]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: @BB.binding.as_type.as.B.impl.%A.type.loc9_34.2 (%A.type.d88d48.1) = name_ref .Self, %.Self.3 [symbolic = %.Self.4 (constants.%.Self.c063ed.1)]
// CHECK:STDOUT:       %.loc9_42.1: @BB.binding.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = specific_constant @X.%assoc0, @A(constants.%AA) [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %X.ref: @BB.binding.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = name_ref X, %.loc9_42.1 [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:       %.loc9_42.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc9_42.1: type = impl_witness_access constants.%A.lookup_impl_witness.ddb77a.1, element0 [symbolic = %impl.elem0.loc9_42.2 (constants.%impl.elem0.7afc4b.1)]
// CHECK:STDOUT:       %.loc9_48.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_48.2: type = converted %.loc9_48.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc9_36.2: type = where_expr %.Self.3 [symbolic = %A_where.type (constants.%A_where.type.ecbf58.1)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%A.type.d88d48.1
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc9_42.1, %.loc9_48.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %BB.loc9_25.1: @BB.binding.as_type.as.B.impl.%A_where.type (%A_where.type.ecbf58.1) = bind_symbolic_name BB, 1 [symbolic = %BB.loc9_25.2 (constants.%BB.afa)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table.loc9 = impl_witness_table (), @BB.binding.as_type.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness.loc9: <witness> = impl_witness %B.impl_witness_table.loc9, @BB.binding.as_type.as.B.impl(constants.%AA, constants.%BB.afa) [symbolic = @BB.binding.as_type.as.B.impl.%B.impl_witness (constants.%B.impl_witness.825)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @BB.binding.as_type.as.B.impl(%AA.loc9_14.1: type, %BB.loc9_25.1: @BB.binding.as_type.as.B.impl.%A_where.type (%A_where.type.ecbf58.1)) {
// CHECK:STDOUT:   %AA.loc9_14.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc9_14.2 (constants.%AA)]
// CHECK:STDOUT:   %A.type.loc9_34.2: type = facet_type <@A, @A(%AA.loc9_14.2)> [symbolic = %A.type.loc9_34.2 (constants.%A.type.d88d48.1)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc9_42: <witness> = require_complete_type %A.type.loc9_34.2 [symbolic = %require_complete.loc9_42 (constants.%require_complete.34a)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc9_14.2) [symbolic = %A.assoc_type (constants.%A.assoc_type)]
// CHECK:STDOUT:   %assoc0: @BB.binding.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = assoc_entity element0, @A.%X [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @A, @A(%AA.loc9_14.2) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.ddb77a.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_42.2: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_42.2 (constants.%impl.elem0.7afc4b.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%AA.loc9_14.2) where %impl.elem0.loc9_42.2 = constants.%empty_tuple.type> [symbolic = %A_where.type (constants.%A_where.type.ecbf58.1)]
// CHECK:STDOUT:   %BB.loc9_25.2: @BB.binding.as_type.as.B.impl.%A_where.type (%A_where.type.ecbf58.1) = bind_symbolic_name BB, 1 [symbolic = %BB.loc9_25.2 (constants.%BB.afa)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.22c)]
// CHECK:STDOUT:   %BB.binding.as_type: type = symbolic_binding_type BB, 1, %BB.loc9_25.2 [symbolic = %BB.binding.as_type (constants.%BB.binding.as_type)]
// CHECK:STDOUT:   %ptr.loc9_61.2: type = ptr_type %AA.loc9_14.2 [symbolic = %ptr.loc9_61.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %B.type.loc9_62.2: type = facet_type <@B, @B(%ptr.loc9_61.2)> [symbolic = %B.type.loc9_62.2 (constants.%B.type.6d756f.1)]
// CHECK:STDOUT:   %require_complete.loc9_62: <witness> = require_complete_type %B.type.loc9_62.2 [symbolic = %require_complete.loc9_62 (constants.%require_complete.672f1b.1)]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness file.%B.impl_witness_table.loc9, @BB.binding.as_type.as.B.impl(%AA.loc9_14.2, %BB.loc9_25.2) [symbolic = %B.impl_witness (constants.%B.impl_witness.825)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_51 as %B.type.loc9_62.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%B.impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc14_9.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.432
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc14_9.2: type, %T.loc15_8.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %ptr.loc21_14.2: type = ptr_type %DD [symbolic = %ptr.loc21_14.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:   %B.type.loc21_15.2: type = facet_type <@B, @B(%ptr.loc21_14.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.6d756f.2)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%DD)> [symbolic = %A.type (constants.%A.type.d88d48.2)]
// CHECK:STDOUT:   %.Self.loc9: @D.G.%A.type (%A.type.d88d48.2) = bind_symbolic_name .Self [symbolic = %.Self.loc9 (constants.%.Self.c063ed.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc9, @A, @A(%DD) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.ddb77a.2)]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.7afc4b.2)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_8.1, @B, @B(%ptr.loc21_14.2) [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.8e3)]
// CHECK:STDOUT:   %B.facet.loc21_7.2: @D.G.%B.type.loc21_15.2 (%B.type.6d756f.2) = facet_value %T.loc15_8.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.c0e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_8.2 [symbolic = %T.loc15_8.1 (constants.%T.336)]
// CHECK:STDOUT:     %B.ref: %B.type.bf2 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc14_9.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %ptr.loc21_14.1: type = ptr_type %DD.ref [symbolic = %ptr.loc21_14.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:     %B.type.loc21_15.1: type = facet_type <@B, @B(constants.%ptr.79f131.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.6d756f.2)]
// CHECK:STDOUT:     %B.facet.loc21_7.1: @D.G.%B.type.loc21_15.2 (%B.type.6d756f.2) = facet_value %T.ref, (constants.%B.lookup_impl_witness.8e3) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.c0e)]
// CHECK:STDOUT:     %.loc21: @D.G.%B.type.loc21_15.2 (%B.type.6d756f.2) = converted %T.ref, %B.facet.loc21_7.1 [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.c0e)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.binding.as_type.as.B.impl(constants.%AA, constants.%BB.afa) {
// CHECK:STDOUT:   %AA.loc9_14.2 => constants.%AA
// CHECK:STDOUT:   %A.type.loc9_34.2 => constants.%A.type.d88d48.1
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.c063ed.1
// CHECK:STDOUT:   %require_complete.loc9_42 => constants.%require_complete.34a
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type
// CHECK:STDOUT:   %assoc0 => constants.%assoc0
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.ddb77a.1
// CHECK:STDOUT:   %impl.elem0.loc9_42.2 => constants.%impl.elem0.7afc4b.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.ecbf58.1
// CHECK:STDOUT:   %BB.loc9_25.2 => constants.%BB.afa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.22c
// CHECK:STDOUT:   %BB.binding.as_type => constants.%BB.binding.as_type
// CHECK:STDOUT:   %ptr.loc9_61.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %B.type.loc9_62.2 => constants.%B.type.6d756f.1
// CHECK:STDOUT:   %require_complete.loc9_62 => constants.%require_complete.672f1b.1
// CHECK:STDOUT:   %B.impl_witness => constants.%B.impl_witness.825
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc14_9.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.a94
// CHECK:STDOUT:   %D.G => constants.%D.G.714
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.336) {
// CHECK:STDOUT:   %T.loc15_8.1 => constants.%T.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc14_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.0da
// CHECK:STDOUT:   %D.G => constants.%D.G.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc15_8.1 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc21_14.2 => constants.%ptr.c28
// CHECK:STDOUT:   %B.type.loc21_15.2 => constants.%B.type.997
// CHECK:STDOUT:   %A.type => constants.%A.type.23f
// CHECK:STDOUT:   %.Self.loc9 => constants.%.Self.495
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.37c
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.e27
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.c4b
// CHECK:STDOUT:   %B.facet.loc21_7.2 => constants.%B.facet.b55
// CHECK:STDOUT: }
// CHECK:STDOUT:
