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
  fn F(unused T:! A(CC) where .X = CC) {}
  //@dump-sem-ir-end
}

class D(DD:! type) {
  fn G(T:! type) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE+7]]:5: error: cannot convert type `T` into type implementing `A(DD) where .(A(DD).X) = DD` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE-10]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(unused T:! A(CC) where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %CC: type = symbolic_binding CC, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1f98dc.1: type = class_type @C, @C(%CC) [symbolic]
// CHECK:STDOUT:   %A.type.5ea944.2: type = facet_type <@A, @A(%CC)> [symbolic]
// CHECK:STDOUT:   %.Self.1b7616.1: %A.type.5ea944.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %A.assoc_type.092091.2: type = assoc_entity_type @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %assoc0.818660.2: %A.assoc_type.092091.2 = assoc_entity element0, @A.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.88d: <witness> = require_complete_type %A.type.5ea944.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.1b7616.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.15e6be.1: <witness> = lookup_impl_witness %.Self.1b7616.1, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b25b5d.1: type = impl_witness_access %A.lookup_impl_witness.15e6be.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.f77576.1: type = facet_type <@A, @A(%CC) where %impl.elem0.b25b5d.1 = %CC> [symbolic]
// CHECK:STDOUT:   %pattern_type.88a: type = pattern_type %A_where.type.f77576.1 [symbolic]
// CHECK:STDOUT:   %T.4f4: %A_where.type.f77576.1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %C.F.type.ac966c.1: type = fn_type @C.F, @C(%CC) [symbolic]
// CHECK:STDOUT:   %C.F.a8a4df.1: %C.F.type.ac966c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic]
// CHECK:STDOUT:   %D.dbf: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.e6f: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.6b0: %D.G.type.e6f = struct_value () [symbolic]
// CHECK:STDOUT:   %C.1f98dc.2: type = class_type @C, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.type.ac966c.2: type = fn_type @C.F, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.a8a4df.2: %C.F.type.ac966c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.122: <witness> = require_complete_type %C.1f98dc.2 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.ffd: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.e3b: %D.G.type.ffd = struct_value () [concrete]
// CHECK:STDOUT:   %C.ea5: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.type.143: type = fn_type @C.F, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.ada: %C.F.type.143 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%CC.loc6_11.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%CC.loc6_11.1) [symbolic = %C.F.type (constants.%C.F.type.ac966c.1)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type.ac966c.1) = struct_value () [symbolic = %C.F (constants.%C.F.a8a4df.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type.ac966c.1) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F.a8a4df.1)] {
// CHECK:STDOUT:       %T.patt: @C.F.%pattern_type (%pattern_type.88a) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_25.1: type = splice_block %.loc11_25.2 [symbolic = %A_where.type (constants.%A_where.type.f77576.1)] {
// CHECK:STDOUT:         %.Self.loc11_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:         %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:         %CC.ref.loc11_21: type = name_ref CC, @C.%CC.loc6_11.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %A.type.loc11_23.2: type = facet_type <@A, @A(constants.%CC)> [symbolic = %A.type.loc11_23.1 (constants.%A.type.5ea944.2)]
// CHECK:STDOUT:         %.Self.loc11_25.2: @C.F.%A.type.loc11_23.1 (%A.type.5ea944.2) = symbolic_binding .Self [symbolic = %.Self.loc11_25.1 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:         %.Self.ref: @C.F.%A.type.loc11_23.1 (%A.type.5ea944.2) = name_ref .Self, %.Self.loc11_25.2 [symbolic = %.Self.loc11_25.1 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:         %.Self.as_type.loc11_31.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc11_31.1 (constants.%.Self.as_type)]
// CHECK:STDOUT:         %.loc11_31.1: type = converted %.Self.ref, %.Self.as_type.loc11_31.2 [symbolic = %.Self.as_type.loc11_31.1 (constants.%.Self.as_type)]
// CHECK:STDOUT:         %.loc11_31.2: @C.F.%A.assoc_type (%A.assoc_type.092091.2) = specific_constant @X.%assoc0, @A.WithSelf(constants.%CC, constants.%.Self.1b7616.1) [symbolic = %assoc0 (constants.%assoc0.818660.2)]
// CHECK:STDOUT:         %X.ref: @C.F.%A.assoc_type (%A.assoc_type.092091.2) = name_ref X, %.loc11_31.2 [symbolic = %assoc0 (constants.%assoc0.818660.2)]
// CHECK:STDOUT:         %impl.elem0.loc11_31.2: type = impl_witness_access constants.%A.lookup_impl_witness.15e6be.1, element0 [symbolic = %impl.elem0.loc11_31.1 (constants.%impl.elem0.b25b5d.1)]
// CHECK:STDOUT:         %CC.ref.loc11_36: type = name_ref CC, @C.%CC.loc6_11.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %.loc11_25.2: type = where_expr [symbolic = %A_where.type (constants.%A_where.type.f77576.1)] {
// CHECK:STDOUT:           requirement_base_facet_type %A.type.loc11_23.2
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc11_31.2, %CC.ref.loc11_36
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc11_16.2: @C.F.%A_where.type (%A_where.type.f77576.1) = symbolic_binding T, 1 [symbolic = %T.loc11_16.1 (constants.%T.4f4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1f98dc.1
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .CC = <poisoned>
// CHECK:STDOUT:     .F = %C.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc15_11.2: type) {
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
// CHECK:STDOUT:     .Self = constants.%D.dbf
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%CC.loc6_11.2: type, %T.loc11_16.2: @C.F.%A_where.type (%A_where.type.f77576.1)) {
// CHECK:STDOUT:   %CC: type = symbolic_binding CC, 0 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:   %A.type.loc11_23.1: type = facet_type <@A, @A(%CC)> [symbolic = %A.type.loc11_23.1 (constants.%A.type.5ea944.2)]
// CHECK:STDOUT:   %.Self.loc11_25.1: @C.F.%A.type.loc11_23.1 (%A.type.5ea944.2) = symbolic_binding .Self [symbolic = %.Self.loc11_25.1 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc11_23.1 [symbolic = %require_complete (constants.%require_complete.88d)]
// CHECK:STDOUT:   %.Self.as_type.loc11_31.1: type = facet_access_type %.Self.loc11_25.1 [symbolic = %.Self.as_type.loc11_31.1 (constants.%.Self.as_type)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%CC) [symbolic = %A.assoc_type (constants.%A.assoc_type.092091.2)]
// CHECK:STDOUT:   %assoc0: @C.F.%A.assoc_type (%A.assoc_type.092091.2) = assoc_entity element0, @A.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.818660.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc11_25.1, @A, @A(%CC) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.15e6be.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_31.1: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_31.1 (constants.%impl.elem0.b25b5d.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%CC) where %impl.elem0.loc11_31.1 = %CC> [symbolic = %A_where.type (constants.%A_where.type.f77576.1)]
// CHECK:STDOUT:   %T.loc11_16.1: @C.F.%A_where.type (%A_where.type.f77576.1) = symbolic_binding T, 1 [symbolic = %T.loc11_16.1 (constants.%T.4f4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.88a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc15_11.2: type, %T.loc16_9.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %C.loc25_9.2: type = class_type @C, @C(%DD) [symbolic = %C.loc25_9.2 (constants.%C.1f98dc.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc25_9.2 [symbolic = %require_complete (constants.%require_complete.122)]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%DD) [symbolic = %C.F.type (constants.%C.F.type.ac966c.2)]
// CHECK:STDOUT:   %C.F: @D.G.%C.F.type (%C.F.type.ac966c.2) = struct_value () [symbolic = %C.F (constants.%C.F.a8a4df.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc15_11.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %C.loc25_9.1: type = class_type @C, @C(constants.%DD) [symbolic = %C.loc25_9.2 (constants.%C.1f98dc.2)]
// CHECK:STDOUT:     %.loc25: @D.G.%C.F.type (%C.F.type.ac966c.2) = specific_constant @C.%C.F.decl, @C(constants.%DD) [symbolic = %C.F (constants.%C.F.a8a4df.2)]
// CHECK:STDOUT:     %F.ref: @D.G.%C.F.type (%C.F.type.ac966c.2) = name_ref F, %.loc25 [symbolic = %C.F (constants.%C.F.a8a4df.2)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_9.2 [symbolic = %T.loc16_9.1 (constants.%T.091)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%CC) {
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.ac966c.1
// CHECK:STDOUT:   %C.F => constants.%C.F.a8a4df.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%CC, constants.%T.4f4) {
// CHECK:STDOUT:   %CC => constants.%CC
// CHECK:STDOUT:   %A.type.loc11_23.1 => constants.%A.type.5ea944.2
// CHECK:STDOUT:   %.Self.loc11_25.1 => constants.%.Self.1b7616.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.88d
// CHECK:STDOUT:   %.Self.as_type.loc11_31.1 => constants.%.Self.as_type
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.092091.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.818660.2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.15e6be.1
// CHECK:STDOUT:   %impl.elem0.loc11_31.1 => constants.%impl.elem0.b25b5d.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.f77576.1
// CHECK:STDOUT:   %T.loc11_16.1 => constants.%T.4f4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc15_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.e6f
// CHECK:STDOUT:   %D.G => constants.%D.G.6b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.091) {
// CHECK:STDOUT:   %T.loc16_9.1 => constants.%T.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%DD) {
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.ac966c.2
// CHECK:STDOUT:   %C.F => constants.%C.F.a8a4df.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc15_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.ffd
// CHECK:STDOUT:   %D.G => constants.%D.G.e3b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.67d) {
// CHECK:STDOUT:   %T.loc16_9.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc25_9.2 => constants.%C.ea5
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.143
// CHECK:STDOUT:   %C.F => constants.%C.F.ada
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.143
// CHECK:STDOUT:   %C.F => constants.%C.F.ada
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dot_self_symbolic_in_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %AA: type = symbolic_binding AA, 0 [symbolic]
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.5ea944.1: type = facet_type <@A, @A(%AA)> [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %B.type.f20: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.1b7616.1: %A.type.5ea944.1 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete.88d: <witness> = require_complete_type %A.type.5ea944.1 [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.1b7616.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.15e6be.1: <witness> = lookup_impl_witness %.Self.1b7616.1, @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b25b5d.1: type = impl_witness_access %A.lookup_impl_witness.15e6be.1, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %A_where.type.3878ef.1: type = facet_type <@A, @A(%AA) where %impl.elem0.b25b5d.1 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.25d: type = pattern_type %A_where.type.3878ef.1 [symbolic]
// CHECK:STDOUT:   %BB.e97: %A_where.type.3878ef.1 = symbolic_binding BB, 1 [symbolic]
// CHECK:STDOUT:   %BB.as_type: type = facet_access_type %BB.e97 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %AA [symbolic]
// CHECK:STDOUT:   %B.type.ef01e2.1: type = facet_type <@B, @B(%ptr.e8f8f9.1)> [symbolic]
// CHECK:STDOUT:   %B.impl_witness.461: <witness> = impl_witness @BB.as_type.as.B.impl.%B.impl_witness_table, @BB.as_type.as.B.impl(%AA, %BB.e97) [symbolic]
// CHECK:STDOUT:   %require_complete.a95288.1: <witness> = require_complete_type %B.type.ef01e2.1 [symbolic]
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic]
// CHECK:STDOUT:   %D.dbf: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.e6f: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.6b0: %D.G.type.e6f = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %DD [symbolic]
// CHECK:STDOUT:   %B.type.ef01e2.2: type = facet_type <@B, @B(%ptr.e8f8f9.2)> [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.332: <witness> = lookup_impl_witness %T.091, @B, @B(%ptr.e8f8f9.2) [symbolic]
// CHECK:STDOUT:   %.9e8: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.e8f8f9.2, %T.091) [symbolic]
// CHECK:STDOUT:   %B.facet.dd5: %B.type.ef01e2.2 = facet_value %T.091, (%B.lookup_impl_witness.332) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.ffd: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.e3b: %D.G.type.ffd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.type.2b3: type = facet_type <@B, @B(%ptr.c28)> [concrete]
// CHECK:STDOUT:   %.878: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.c28, %T.67d) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.6d2: <witness> = lookup_impl_witness %T.67d, @B, @B(%ptr.c28) [symbolic]
// CHECK:STDOUT:   %B.facet.a5c: %B.type.2b3 = facet_value %T.67d, (%B.lookup_impl_witness.6d2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @BB.as_type.as.B.impl [concrete] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %BB.patt: @BB.as_type.as.B.impl.%pattern_type (%pattern_type.25d) = symbolic_binding_pattern BB, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %BB.ref: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.3878ef.1) = name_ref BB, %BB.loc9_27.1 [symbolic = %BB.loc9_27.2 (constants.%BB.e97)]
// CHECK:STDOUT:     %BB.as_type.loc9_51.1: type = facet_access_type %BB.ref [symbolic = %BB.as_type.loc9_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %.loc9_51: type = converted %BB.ref, %BB.as_type.loc9_51.1 [symbolic = %BB.as_type.loc9_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %B.ref: %B.type.f20 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %AA.ref.loc9_59: type = name_ref AA, %AA.loc9_16.1 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:     %ptr.loc9_61.1: type = ptr_type %AA.ref.loc9_59 [symbolic = %ptr.loc9_61.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %B.type.loc9_62.1: type = facet_type <@B, @B(constants.%ptr.e8f8f9.1)> [symbolic = %B.type.loc9_62.2 (constants.%B.type.ef01e2.1)]
// CHECK:STDOUT:     %.loc9_19.1: type = splice_block %.loc9_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc9_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AA.loc9_16.1: type = symbolic_binding AA, 0 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:     %.loc9_36.1: type = splice_block %.loc9_36.2 [symbolic = %A_where.type (constants.%A_where.type.3878ef.1)] {
// CHECK:STDOUT:       %.Self.loc9_27: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %AA.ref.loc9_32: type = name_ref AA, %AA.loc9_16.1 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:       %A.type.loc9_34.1: type = facet_type <@A, @A(constants.%AA)> [symbolic = %A.type.loc9_34.2 (constants.%A.type.5ea944.1)]
// CHECK:STDOUT:       %.Self.loc9_36.1: @BB.as_type.as.B.impl.%A.type.loc9_34.2 (%A.type.5ea944.1) = symbolic_binding .Self [symbolic = %.Self.loc9_36.2 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:       %.Self.ref: @BB.as_type.as.B.impl.%A.type.loc9_34.2 (%A.type.5ea944.1) = name_ref .Self, %.Self.loc9_36.1 [symbolic = %.Self.loc9_36.2 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:       %.Self.as_type.loc9_42.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc9_42.2 (constants.%.Self.as_type)]
// CHECK:STDOUT:       %.loc9_42.1: type = converted %.Self.ref, %.Self.as_type.loc9_42.1 [symbolic = %.Self.as_type.loc9_42.2 (constants.%.Self.as_type)]
// CHECK:STDOUT:       %.loc9_42.2: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = specific_constant @X.%assoc0, @A.WithSelf(constants.%AA, constants.%.Self.1b7616.1) [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %X.ref: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = name_ref X, %.loc9_42.2 [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %impl.elem0.loc9_42.1: type = impl_witness_access constants.%A.lookup_impl_witness.15e6be.1, element0 [symbolic = %impl.elem0.loc9_42.2 (constants.%impl.elem0.b25b5d.1)]
// CHECK:STDOUT:       %.loc9_48.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_48.2: type = converted %.loc9_48.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc9_36.2: type = where_expr [symbolic = %A_where.type (constants.%A_where.type.3878ef.1)] {
// CHECK:STDOUT:         requirement_base_facet_type %A.type.loc9_34.1
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc9_42.1, %.loc9_48.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %BB.loc9_27.1: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.3878ef.1) = symbolic_binding BB, 1 [symbolic = %BB.loc9_27.2 (constants.%BB.e97)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @BB.as_type.as.B.impl(%AA.loc9_16.1: type, %BB.loc9_27.1: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.3878ef.1)) {
// CHECK:STDOUT:   %AA.loc9_16.2: type = symbolic_binding AA, 0 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:   %A.type.loc9_34.2: type = facet_type <@A, @A(%AA.loc9_16.2)> [symbolic = %A.type.loc9_34.2 (constants.%A.type.5ea944.1)]
// CHECK:STDOUT:   %.Self.loc9_36.2: @BB.as_type.as.B.impl.%A.type.loc9_34.2 (%A.type.5ea944.1) = symbolic_binding .Self [symbolic = %.Self.loc9_36.2 (constants.%.Self.1b7616.1)]
// CHECK:STDOUT:   %require_complete.loc9_42: <witness> = require_complete_type %A.type.loc9_34.2 [symbolic = %require_complete.loc9_42 (constants.%require_complete.88d)]
// CHECK:STDOUT:   %.Self.as_type.loc9_42.2: type = facet_access_type %.Self.loc9_36.2 [symbolic = %.Self.as_type.loc9_42.2 (constants.%.Self.as_type)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc9_16.2) [symbolic = %A.assoc_type (constants.%A.assoc_type)]
// CHECK:STDOUT:   %assoc0: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = assoc_entity element0, @A.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc9_36.2, @A, @A(%AA.loc9_16.2) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.15e6be.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_42.2: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_42.2 (constants.%impl.elem0.b25b5d.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%AA.loc9_16.2) where %impl.elem0.loc9_42.2 = constants.%empty_tuple.type> [symbolic = %A_where.type (constants.%A_where.type.3878ef.1)]
// CHECK:STDOUT:   %BB.loc9_27.2: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.3878ef.1) = symbolic_binding BB, 1 [symbolic = %BB.loc9_27.2 (constants.%BB.e97)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.25d)]
// CHECK:STDOUT:   %BB.as_type.loc9_51.2: type = facet_access_type %BB.loc9_27.2 [symbolic = %BB.as_type.loc9_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:   %ptr.loc9_61.2: type = ptr_type %AA.loc9_16.2 [symbolic = %ptr.loc9_61.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %B.type.loc9_62.2: type = facet_type <@B, @B(%ptr.loc9_61.2)> [symbolic = %B.type.loc9_62.2 (constants.%B.type.ef01e2.1)]
// CHECK:STDOUT:   %B.impl_witness.loc9_64.2: <witness> = impl_witness %B.impl_witness_table, @BB.as_type.as.B.impl(%AA.loc9_16.2, %BB.loc9_27.2) [symbolic = %B.impl_witness.loc9_64.2 (constants.%B.impl_witness.461)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_62: <witness> = require_complete_type %B.type.loc9_62.2 [symbolic = %require_complete.loc9_62 (constants.%require_complete.a95288.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_51 as %B.type.loc9_62.1 {
// CHECK:STDOUT:     %B.impl_witness_table = impl_witness_table (), @BB.as_type.as.B.impl [concrete]
// CHECK:STDOUT:     %B.impl_witness.loc9_64.1: <witness> = impl_witness %B.impl_witness_table, @BB.as_type.as.B.impl(constants.%AA, constants.%BB.e97) [symbolic = %B.impl_witness.loc9_64.2 (constants.%B.impl_witness.461)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %B.type.loc9_62.1
// CHECK:STDOUT:     witness = %B.impl_witness.loc9_64.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc14_11.2: type) {
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
// CHECK:STDOUT:     .Self = constants.%D.dbf
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc14_11.2: type, %T.loc15_9.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %ptr.loc21_14.2: type = ptr_type %DD [symbolic = %ptr.loc21_14.2 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %B.type.loc21_15.2: type = facet_type <@B, @B(%ptr.loc21_14.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.ef01e2.2)]
// CHECK:STDOUT:   %.loc21_7.2: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.loc21_14.2, %T.loc15_9.1) [symbolic = %.loc21_7.2 (constants.%.9e8)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_9.1, @B, @B(%ptr.loc21_14.2) [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.332)]
// CHECK:STDOUT:   %B.facet.loc21_7.2: @D.G.%B.type.loc21_15.2 (%B.type.ef01e2.2) = facet_value %T.loc15_9.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.dd5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_9.2 [symbolic = %T.loc15_9.1 (constants.%T.091)]
// CHECK:STDOUT:     %B.ref: %B.type.f20 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc14_11.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %ptr.loc21_14.1: type = ptr_type %DD.ref [symbolic = %ptr.loc21_14.2 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %B.type.loc21_15.1: type = facet_type <@B, @B(constants.%ptr.e8f8f9.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.ef01e2.2)]
// CHECK:STDOUT:     %B.facet.loc21_7.1: @D.G.%B.type.loc21_15.2 (%B.type.ef01e2.2) = facet_value %T.ref, (constants.%B.lookup_impl_witness.332) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.dd5)]
// CHECK:STDOUT:     %.loc21_7.1: @D.G.%B.type.loc21_15.2 (%B.type.ef01e2.2) = converted %T.ref, %B.facet.loc21_7.1 [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.dd5)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.as_type.as.B.impl(constants.%AA, constants.%BB.e97) {
// CHECK:STDOUT:   %AA.loc9_16.2 => constants.%AA
// CHECK:STDOUT:   %A.type.loc9_34.2 => constants.%A.type.5ea944.1
// CHECK:STDOUT:   %.Self.loc9_36.2 => constants.%.Self.1b7616.1
// CHECK:STDOUT:   %require_complete.loc9_42 => constants.%require_complete.88d
// CHECK:STDOUT:   %.Self.as_type.loc9_42.2 => constants.%.Self.as_type
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type
// CHECK:STDOUT:   %assoc0 => constants.%assoc0
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.15e6be.1
// CHECK:STDOUT:   %impl.elem0.loc9_42.2 => constants.%impl.elem0.b25b5d.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.3878ef.1
// CHECK:STDOUT:   %BB.loc9_27.2 => constants.%BB.e97
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.25d
// CHECK:STDOUT:   %BB.as_type.loc9_51.2 => constants.%BB.as_type
// CHECK:STDOUT:   %ptr.loc9_61.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %B.type.loc9_62.2 => constants.%B.type.ef01e2.1
// CHECK:STDOUT:   %B.impl_witness.loc9_64.2 => constants.%B.impl_witness.461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc14_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.e6f
// CHECK:STDOUT:   %D.G => constants.%D.G.6b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.091) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%T.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc14_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.ffd
// CHECK:STDOUT:   %D.G => constants.%D.G.e3b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.67d) {
// CHECK:STDOUT:   %T.loc15_9.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc21_14.2 => constants.%ptr.c28
// CHECK:STDOUT:   %B.type.loc21_15.2 => constants.%B.type.2b3
// CHECK:STDOUT:   %.loc21_7.2 => constants.%.878
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.6d2
// CHECK:STDOUT:   %B.facet.loc21_7.2 => constants.%B.facet.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
