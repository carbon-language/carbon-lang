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

interface A(AA: type) { let X: type; }

class C(CC: type) {
  //@dump-sem-ir-begin
  // The `.Self` bind_symbolic_name here should have a
  // symbolic constant value, not symbolic_self, because
  // its type depends on `CC`.
  fn F(unused generic T: A(CC) where .X = CC) {}
  //@dump-sem-ir-end
}

class D(DD: type) {
  fn G(generic T: type) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE+7]]:5: error: cannot convert type `T` into type implementing `A(DD) where .(A(DD).X) = DD` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE-10]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(unused generic T: A(CC) where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
    //@dump-sem-ir-end
  }
}

fn H(generic T: type) {
  D({}).G(T);
}

// --- dot_self_symbolic_in_impl_lookup.carbon

library "[[@TEST_NAME]]";

interface A(AA: type) { let X: type; }

interface B(AA: type) {}

//@dump-sem-ir-begin
impl forall [AA: type, BB: A(AA) where .X = ()] BB as B(AA*) {}
//@dump-sem-ir-end

impl forall [AA: type, BB: type] BB as B(AA) {}

class D(DD: type) {
  fn G(generic T: type) {
    // TODO: This produces a `lookup_impl_witness` instruction looking
    // for `.Self` as `A(DD)` in the generic eval block for `G`. That
    // is from an impl we ended up not using, so shouldn't be included
    // in the eval block.
    //@dump-sem-ir-begin
    T as B(DD*);
    //@dump-sem-ir-end
  }
}

fn H(generic T: type) {
  D({}).G(T);
}

// CHECK:STDOUT: --- fail_dot_self_symbolic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %CC.patt: %pattern_type.9a5 = symbolic_binding_pattern CC, 0 [symbolic]
// CHECK:STDOUT:   %CC: type = symbolic_binding CC, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.6bcb08.1: type = class_type @C, @C(%CC) [symbolic]
// CHECK:STDOUT:   %A.type.fa003b.2: type = facet_type <@A, @A(%CC)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.b15: %A.type.fa003b.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %A.assoc_type.3230b8.2: type = assoc_entity_type @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %assoc0.4c905d.2: %A.assoc_type.3230b8.2 = assoc_entity element0, @A.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.792: <witness> = require_complete_type %A.type.fa003b.2 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.b15 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.030: <witness> = lookup_impl_witness %.Self.frozen.b15, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %impl.elem0.fba: type = impl_witness_access %A.lookup_impl_witness.030, element0 [symbolic]
// CHECK:STDOUT:   %.Self.8ed243.1: %A.type.fa003b.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.9e772d.1: <witness> = lookup_impl_witness %.Self.8ed243.1, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7f79de.1: type = impl_witness_access %A.lookup_impl_witness.9e772d.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.ff151a.1: type = facet_type <@A, @A(%CC) where %impl.elem0.7f79de.1 = %CC> [symbolic]
// CHECK:STDOUT:   %pattern_type.957: type = pattern_type %A_where.type.ff151a.1 [symbolic]
// CHECK:STDOUT:   %T.patt.9e9: %pattern_type.957 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.5e2: %A_where.type.ff151a.1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %C.F.type.ba1cca.1: type = fn_type @C.F, @C(%CC) [symbolic]
// CHECK:STDOUT:   %C.F.e00d14.1: %C.F.type.ba1cca.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DD.patt: %pattern_type.9a5 = symbolic_binding_pattern DD, 0 [symbolic]
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic]
// CHECK:STDOUT:   %D.743: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.patt.cb8: %pattern_type.9a5 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.cc0: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.6b9: %D.G.type.cc0 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.6bcb08.2: type = class_type @C, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.type.ba1cca.2: type = fn_type @C.F, @C(%DD) [symbolic]
// CHECK:STDOUT:   %C.F.e00d14.2: %C.F.type.ba1cca.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.3f6: <witness> = require_complete_type %C.6bcb08.2 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.bec: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.50b: %D.G.type.bec = struct_value () [concrete]
// CHECK:STDOUT:   %C.540: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.type.a0d: type = fn_type @C.F, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.F.2d7: %C.F.type.a0d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%CC.loc6_11.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%CC.loc6_11.1) [symbolic = %C.F.type (constants.%C.F.type.ba1cca.1)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type.ba1cca.1) = struct_value () [symbolic = %C.F (constants.%C.F.e00d14.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type.ba1cca.1) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F.e00d14.1)] {
// CHECK:STDOUT:       %T.patt.loc11_24.1: @C.F.%pattern_type (%pattern_type.957) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc11_24.2 (constants.%T.patt.9e9)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_32.1: type = splice_block %.loc11_32.2 [symbolic = %A_where.type (constants.%A_where.type.ff151a.1)] {
// CHECK:STDOUT:         %.Self.frozen.loc11_24: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:         %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:         %CC.ref.loc11_28: type = name_ref CC, @C.%CC.loc6_11.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %A.type.loc11_30.2: type = facet_type <@A, @A(constants.%CC)> [symbolic = %A.type.loc11_30.1 (constants.%A.type.fa003b.2)]
// CHECK:STDOUT:         %.Self.frozen.loc11_32.2: @C.F.%A.type.loc11_30.1 (%A.type.fa003b.2) = symbolic_binding .Self [symbolic = %.Self.frozen.loc11_32.1 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %A.type.loc11_30.2 [concrete]
// CHECK:STDOUT:         %.Self.ref: @C.F.%A.type.loc11_30.1 (%A.type.fa003b.2) = name_ref .Self, %.Self.frozen.loc11_32.2 [symbolic = %.Self.frozen.loc11_32.1 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:         %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:         %.loc11_38.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:         %.loc11_38.2: @C.F.%A.assoc_type (%A.assoc_type.3230b8.2) = specific_constant @X.%assoc0, @A.WithSelf(constants.%CC, constants.%.Self.frozen.b15) [symbolic = %assoc0 (constants.%assoc0.4c905d.2)]
// CHECK:STDOUT:         %X.ref: @C.F.%A.assoc_type (%A.assoc_type.3230b8.2) = name_ref X, %.loc11_38.2 [symbolic = %assoc0 (constants.%assoc0.4c905d.2)]
// CHECK:STDOUT:         %impl.elem0.loc11_38.3: type = impl_witness_access constants.%A.lookup_impl_witness.030, element0 [symbolic = %impl.elem0.loc11_38.1 (constants.%impl.elem0.fba)]
// CHECK:STDOUT:         %CC.ref.loc11_43: type = name_ref CC, @C.%CC.loc6_11.2 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %rewrite.loc11_41.1 = requirement_rewrite %impl.elem0.loc11_38.3, %CC.ref.loc11_43 [concrete]
// CHECK:STDOUT:         %impl.elem0.loc11_38.4: type = impl_witness_access constants.%A.lookup_impl_witness.9e772d.1, element0 [symbolic = %impl.elem0.loc11_38.2 (constants.%impl.elem0.7f79de.1)]
// CHECK:STDOUT:         %rewrite.loc11_41.2 = requirement_rewrite %impl.elem0.loc11_38.4, %CC.ref.loc11_43 [concrete]
// CHECK:STDOUT:         %.loc11_32.2: type = where_expr [symbolic = %A_where.type (constants.%A_where.type.ff151a.1)] {
// CHECK:STDOUT:           %base_facet_type = requirement_base_facet_type %A.type.loc11_30.2 [concrete]
// CHECK:STDOUT:           %rewrite.loc11_41.2 = requirement_rewrite %impl.elem0.loc11_38.4, %CC.ref.loc11_43 [concrete]
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc11_24.2: @C.F.%A_where.type (%A_where.type.ff151a.1) = symbolic_binding T, 1 [symbolic = %T.loc11_24.1 (constants.%T.5e2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.6bcb08.1
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
// CHECK:STDOUT:     .Self = constants.%D.743
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%CC.loc6_11.2: type, %T.loc11_24.2: @C.F.%A_where.type (%A_where.type.ff151a.1)) {
// CHECK:STDOUT:   %CC: type = symbolic_binding CC, 0 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:   %A.type.loc11_30.1: type = facet_type <@A, @A(%CC)> [symbolic = %A.type.loc11_30.1 (constants.%A.type.fa003b.2)]
// CHECK:STDOUT:   %.Self.frozen.loc11_32.1: @C.F.%A.type.loc11_30.1 (%A.type.fa003b.2) = symbolic_binding .Self [symbolic = %.Self.frozen.loc11_32.1 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc11_30.1 [symbolic = %require_complete (constants.%require_complete.792)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc11_32.1 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%CC) [symbolic = %A.assoc_type (constants.%A.assoc_type.3230b8.2)]
// CHECK:STDOUT:   %assoc0: @C.F.%A.assoc_type (%A.assoc_type.3230b8.2) = assoc_entity element0, @A.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.4c905d.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness.loc11_38.1: <witness> = lookup_impl_witness %.Self.frozen.loc11_32.1, @A, @A(%CC) [symbolic = %A.lookup_impl_witness.loc11_38.1 (constants.%A.lookup_impl_witness.030)]
// CHECK:STDOUT:   %impl.elem0.loc11_38.1: type = impl_witness_access %A.lookup_impl_witness.loc11_38.1, element0 [symbolic = %impl.elem0.loc11_38.1 (constants.%impl.elem0.fba)]
// CHECK:STDOUT:   %.Self: @C.F.%A.type.loc11_30.1 (%A.type.fa003b.2) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.8ed243.1)]
// CHECK:STDOUT:   %A.lookup_impl_witness.loc11_38.2: <witness> = lookup_impl_witness %.Self, @A, @A(%CC) [symbolic = %A.lookup_impl_witness.loc11_38.2 (constants.%A.lookup_impl_witness.9e772d.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_38.2: type = impl_witness_access %A.lookup_impl_witness.loc11_38.2, element0 [symbolic = %impl.elem0.loc11_38.2 (constants.%impl.elem0.7f79de.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%CC) where %impl.elem0.loc11_38.2 = %CC> [symbolic = %A_where.type (constants.%A_where.type.ff151a.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.957)]
// CHECK:STDOUT:   %T.patt.loc11_24.2: @C.F.%pattern_type (%pattern_type.957) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc11_24.2 (constants.%T.patt.9e9)]
// CHECK:STDOUT:   %T.loc11_24.1: @C.F.%A_where.type (%A_where.type.ff151a.1) = symbolic_binding T, 1 [symbolic = %T.loc11_24.1 (constants.%T.5e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc15_11.2: type, %T.loc16_17.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %C.loc25_9.2: type = class_type @C, @C(%DD) [symbolic = %C.loc25_9.2 (constants.%C.6bcb08.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc25_9.2 [symbolic = %require_complete (constants.%require_complete.3f6)]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%DD) [symbolic = %C.F.type (constants.%C.F.type.ba1cca.2)]
// CHECK:STDOUT:   %C.F: @D.G.%C.F.type (%C.F.type.ba1cca.2) = struct_value () [symbolic = %C.F (constants.%C.F.e00d14.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc15_11.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %C.loc25_9.1: type = class_type @C, @C(constants.%DD) [symbolic = %C.loc25_9.2 (constants.%C.6bcb08.2)]
// CHECK:STDOUT:     %.loc25: @D.G.%C.F.type (%C.F.type.ba1cca.2) = specific_constant @C.%C.F.decl, @C(constants.%DD) [symbolic = %C.F (constants.%C.F.e00d14.2)]
// CHECK:STDOUT:     %F.ref: @D.G.%C.F.type (%C.F.type.ba1cca.2) = name_ref F, %.loc25 [symbolic = %C.F (constants.%C.F.e00d14.2)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_17.2 [symbolic = %T.loc16_17.1 (constants.%T.091)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%CC) {
// CHECK:STDOUT:   %CC.patt.loc6_11.2 => constants.%CC.patt
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.ba1cca.1
// CHECK:STDOUT:   %C.F => constants.%C.F.e00d14.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%CC, constants.%T.5e2) {
// CHECK:STDOUT:   %CC => constants.%CC
// CHECK:STDOUT:   %A.type.loc11_30.1 => constants.%A.type.fa003b.2
// CHECK:STDOUT:   %.Self.frozen.loc11_32.1 => constants.%.Self.frozen.b15
// CHECK:STDOUT:   %require_complete => constants.%require_complete.792
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3230b8.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4c905d.2
// CHECK:STDOUT:   %A.lookup_impl_witness.loc11_38.1 => constants.%A.lookup_impl_witness.030
// CHECK:STDOUT:   %impl.elem0.loc11_38.1 => constants.%impl.elem0.fba
// CHECK:STDOUT:   %.Self => constants.%.Self.8ed243.1
// CHECK:STDOUT:   %A.lookup_impl_witness.loc11_38.2 => constants.%A.lookup_impl_witness.9e772d.1
// CHECK:STDOUT:   %impl.elem0.loc11_38.2 => constants.%impl.elem0.7f79de.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.ff151a.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.957
// CHECK:STDOUT:   %T.patt.loc11_24.2 => constants.%T.patt.9e9
// CHECK:STDOUT:   %T.loc11_24.1 => constants.%T.5e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.patt.loc15_11.2 => constants.%DD.patt
// CHECK:STDOUT:   %DD.loc15_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.cc0
// CHECK:STDOUT:   %D.G => constants.%D.G.6b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.091) {
// CHECK:STDOUT:   %T.patt.loc16_17.2 => constants.%T.patt.cb8
// CHECK:STDOUT:   %T.loc16_17.1 => constants.%T.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%DD) {
// CHECK:STDOUT:   %CC.patt.loc6_11.2 => constants.%CC.patt
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.ba1cca.2
// CHECK:STDOUT:   %C.F => constants.%C.F.e00d14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.patt.loc15_11.2 => constants.%DD.patt
// CHECK:STDOUT:   %DD.loc15_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.bec
// CHECK:STDOUT:   %D.G => constants.%D.G.50b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc16_17.2 => constants.%T.patt.cb8
// CHECK:STDOUT:   %T.loc16_17.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc25_9.2 => constants.%C.540
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.a0d
// CHECK:STDOUT:   %C.F => constants.%C.F.2d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %CC.patt.loc6_11.2 => constants.%CC.patt
// CHECK:STDOUT:   %CC.loc6_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type.a0d
// CHECK:STDOUT:   %C.F => constants.%C.F.2d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dot_self_symbolic_in_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %AA.patt: %pattern_type.9a5 = symbolic_binding_pattern AA, 0 [symbolic]
// CHECK:STDOUT:   %AA: type = symbolic_binding AA, 0 [symbolic]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.fa003b.1: type = facet_type <@A, @A(%AA)> [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %B.type.d1b: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.d1b = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.b15: %A.type.fa003b.1 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete.792: <witness> = require_complete_type %A.type.fa003b.1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.b15 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.030: <witness> = lookup_impl_witness %.Self.frozen.b15, @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %impl.elem0.fba: type = impl_witness_access %A.lookup_impl_witness.030, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.8ed243.1: %A.type.fa003b.1 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.9e772d.1: <witness> = lookup_impl_witness %.Self.8ed243.1, @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7f79de.1: type = impl_witness_access %A.lookup_impl_witness.9e772d.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.83dbed.1: type = facet_type <@A, @A(%AA) where %impl.elem0.7f79de.1 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.9b5: type = pattern_type %A_where.type.83dbed.1 [symbolic]
// CHECK:STDOUT:   %BB.patt.954: %pattern_type.9b5 = symbolic_binding_pattern BB, 1 [symbolic]
// CHECK:STDOUT:   %BB.a11: %A_where.type.83dbed.1 = symbolic_binding BB, 1 [symbolic]
// CHECK:STDOUT:   %BB.as_type: type = facet_access_type %BB.a11 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %AA [symbolic]
// CHECK:STDOUT:   %B.type.904f2f.1: type = facet_type <@B, @B(%ptr.e8f8f9.1)> [symbolic]
// CHECK:STDOUT:   %.61f: <witness> = impl_self_witness %BB.a11, @B, @B(%ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %B.impl_witness.089: <witness> = impl_witness @BB.as_type.as.B.impl.%B.impl_witness_table, @BB.as_type.as.B.impl(%AA, %BB.a11) [symbolic]
// CHECK:STDOUT:   %require_complete.58dcfe.1: <witness> = require_complete_type %B.type.904f2f.1 [symbolic]
// CHECK:STDOUT:   %DD.patt: %pattern_type.9a5 = symbolic_binding_pattern DD, 0 [symbolic]
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic]
// CHECK:STDOUT:   %D.743: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.patt.cb8: %pattern_type.9a5 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %D.G.type.cc0: type = fn_type @D.G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %D.G.6b9: %D.G.type.cc0 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %DD [symbolic]
// CHECK:STDOUT:   %B.type.904f2f.2: type = facet_type <@B, @B(%ptr.e8f8f9.2)> [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.ab0: <witness> = lookup_impl_witness %T.091, @B, @B(%ptr.e8f8f9.2) [symbolic]
// CHECK:STDOUT:   %.5dd: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.e8f8f9.2, %T.091) [symbolic]
// CHECK:STDOUT:   %B.facet.ed9: %B.type.904f2f.2 = facet_value %T.091, (%B.lookup_impl_witness.ab0) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.G.type.bec: type = fn_type @D.G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %D.G.50b: %D.G.type.bec = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.type.5e6: type = facet_type <@B, @B(%ptr.c28)> [concrete]
// CHECK:STDOUT:   %.e6e: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.c28, %T.67d) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.295: <witness> = lookup_impl_witness %T.67d, @B, @B(%ptr.c28) [symbolic]
// CHECK:STDOUT:   %B.facet.b2d: %B.type.5e6 = facet_value %T.67d, (%B.lookup_impl_witness.295) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @BB.as_type.as.B.impl [concrete] {
// CHECK:STDOUT:     %AA.patt.loc9_16.1: %pattern_type.9a5 = symbolic_binding_pattern AA, 0 [symbolic = %AA.patt.loc9_16.2 (constants.%AA.patt)]
// CHECK:STDOUT:     %BB.patt.loc9_26.1: @BB.as_type.as.B.impl.%pattern_type (%pattern_type.9b5) = symbolic_binding_pattern BB, 1 [symbolic = %BB.patt.loc9_26.2 (constants.%BB.patt.954)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %BB.ref: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.83dbed.1) = name_ref BB, %BB.loc9_26.1 [symbolic = %BB.loc9_26.2 (constants.%BB.a11)]
// CHECK:STDOUT:     %BB.as_type.loc9_49.1: type = facet_access_type %BB.ref [symbolic = %BB.as_type.loc9_49.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %.loc9_49: type = converted %BB.ref, %BB.as_type.loc9_49.1 [symbolic = %BB.as_type.loc9_49.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %B.ref: %B.type.d1b = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %AA.ref.loc9_57: type = name_ref AA, %AA.loc9_16.1 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:     %ptr.loc9_59.1: type = ptr_type %AA.ref.loc9_57 [symbolic = %ptr.loc9_59.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %B.type.loc9_60.1: type = facet_type <@B, @B(constants.%ptr.e8f8f9.1)> [symbolic = %B.type.loc9_60.2 (constants.%B.type.904f2f.1)]
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_16: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AA.loc9_16.1: type = symbolic_binding AA, 0 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:     %.loc9_34.1: type = splice_block %.loc9_34.2 [symbolic = %A_where.type (constants.%A_where.type.83dbed.1)] {
// CHECK:STDOUT:       %.Self.frozen.loc9_26: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %AA.ref.loc9_30: type = name_ref AA, %AA.loc9_16.1 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:       %A.type.loc9_32.1: type = facet_type <@A, @A(constants.%AA)> [symbolic = %A.type.loc9_32.2 (constants.%A.type.fa003b.1)]
// CHECK:STDOUT:       %.Self.frozen.loc9_34.1: @BB.as_type.as.B.impl.%A.type.loc9_32.2 (%A.type.fa003b.1) = symbolic_binding .Self [symbolic = %.Self.frozen.loc9_34.2 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %A.type.loc9_32.1 [concrete]
// CHECK:STDOUT:       %.Self.ref: @BB.as_type.as.B.impl.%A.type.loc9_32.2 (%A.type.fa003b.1) = name_ref .Self, %.Self.frozen.loc9_34.1 [symbolic = %.Self.frozen.loc9_34.2 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:       %.loc9_40.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:       %.loc9_40.2: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = specific_constant @X.%assoc0, @A.WithSelf(constants.%AA, constants.%.Self.frozen.b15) [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %X.ref: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = name_ref X, %.loc9_40.2 [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:       %impl.elem0.loc9_40.1: type = impl_witness_access constants.%A.lookup_impl_witness.030, element0 [symbolic = %impl.elem0.loc9_40.3 (constants.%impl.elem0.fba)]
// CHECK:STDOUT:       %.loc9_46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_46.2: type = converted %.loc9_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc9_43.1 = requirement_rewrite %impl.elem0.loc9_40.1, %.loc9_46.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc9_40.2: type = impl_witness_access constants.%A.lookup_impl_witness.9e772d.1, element0 [symbolic = %impl.elem0.loc9_40.4 (constants.%impl.elem0.7f79de.1)]
// CHECK:STDOUT:       %rewrite.loc9_43.2 = requirement_rewrite %impl.elem0.loc9_40.2, %.loc9_46.2 [concrete]
// CHECK:STDOUT:       %.loc9_34.2: type = where_expr [symbolic = %A_where.type (constants.%A_where.type.83dbed.1)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %A.type.loc9_32.1 [concrete]
// CHECK:STDOUT:         %rewrite.loc9_43.2 = requirement_rewrite %impl.elem0.loc9_40.2, %.loc9_46.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %BB.loc9_26.1: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.83dbed.1) = symbolic_binding BB, 1 [symbolic = %BB.loc9_26.2 (constants.%BB.a11)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @BB.as_type.as.B.impl.%.loc9_49, @B, @B(constants.%ptr.e8f8f9.1) [symbolic = @BB.as_type.as.B.impl.%.loc9_62 (constants.%.61f)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @BB.as_type.as.B.impl(%AA.loc9_16.1: type, %BB.loc9_26.1: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.83dbed.1)) {
// CHECK:STDOUT:   %AA.patt.loc9_16.2: %pattern_type.9a5 = symbolic_binding_pattern AA, 0 [symbolic = %AA.patt.loc9_16.2 (constants.%AA.patt)]
// CHECK:STDOUT:   %AA.loc9_16.2: type = symbolic_binding AA, 0 [symbolic = %AA.loc9_16.2 (constants.%AA)]
// CHECK:STDOUT:   %A.type.loc9_32.2: type = facet_type <@A, @A(%AA.loc9_16.2)> [symbolic = %A.type.loc9_32.2 (constants.%A.type.fa003b.1)]
// CHECK:STDOUT:   %.Self.frozen.loc9_34.2: @BB.as_type.as.B.impl.%A.type.loc9_32.2 (%A.type.fa003b.1) = symbolic_binding .Self [symbolic = %.Self.frozen.loc9_34.2 (constants.%.Self.frozen.b15)]
// CHECK:STDOUT:   %require_complete.loc9_40: <witness> = require_complete_type %A.type.loc9_32.2 [symbolic = %require_complete.loc9_40 (constants.%require_complete.792)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc9_34.2 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc9_16.2) [symbolic = %A.assoc_type (constants.%A.assoc_type)]
// CHECK:STDOUT:   %assoc0: @BB.as_type.as.B.impl.%A.assoc_type (%A.assoc_type) = assoc_entity element0, @A.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:   %A.lookup_impl_witness.loc9_40.1: <witness> = lookup_impl_witness %.Self.frozen.loc9_34.2, @A, @A(%AA.loc9_16.2) [symbolic = %A.lookup_impl_witness.loc9_40.1 (constants.%A.lookup_impl_witness.030)]
// CHECK:STDOUT:   %impl.elem0.loc9_40.3: type = impl_witness_access %A.lookup_impl_witness.loc9_40.1, element0 [symbolic = %impl.elem0.loc9_40.3 (constants.%impl.elem0.fba)]
// CHECK:STDOUT:   %.Self: @BB.as_type.as.B.impl.%A.type.loc9_32.2 (%A.type.fa003b.1) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.8ed243.1)]
// CHECK:STDOUT:   %A.lookup_impl_witness.loc9_40.2: <witness> = lookup_impl_witness %.Self, @A, @A(%AA.loc9_16.2) [symbolic = %A.lookup_impl_witness.loc9_40.2 (constants.%A.lookup_impl_witness.9e772d.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_40.4: type = impl_witness_access %A.lookup_impl_witness.loc9_40.2, element0 [symbolic = %impl.elem0.loc9_40.4 (constants.%impl.elem0.7f79de.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%AA.loc9_16.2) where %impl.elem0.loc9_40.4 = constants.%empty_tuple.type> [symbolic = %A_where.type (constants.%A_where.type.83dbed.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.9b5)]
// CHECK:STDOUT:   %BB.patt.loc9_26.2: @BB.as_type.as.B.impl.%pattern_type (%pattern_type.9b5) = symbolic_binding_pattern BB, 1 [symbolic = %BB.patt.loc9_26.2 (constants.%BB.patt.954)]
// CHECK:STDOUT:   %BB.loc9_26.2: @BB.as_type.as.B.impl.%A_where.type (%A_where.type.83dbed.1) = symbolic_binding BB, 1 [symbolic = %BB.loc9_26.2 (constants.%BB.a11)]
// CHECK:STDOUT:   %BB.as_type.loc9_49.2: type = facet_access_type %BB.loc9_26.2 [symbolic = %BB.as_type.loc9_49.2 (constants.%BB.as_type)]
// CHECK:STDOUT:   %ptr.loc9_59.2: type = ptr_type %AA.loc9_16.2 [symbolic = %ptr.loc9_59.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %B.type.loc9_60.2: type = facet_type <@B, @B(%ptr.loc9_59.2)> [symbolic = %B.type.loc9_60.2 (constants.%B.type.904f2f.1)]
// CHECK:STDOUT:   %.loc9_62: <witness> = impl_self_witness %BB.loc9_26.2, @B, @B(%ptr.loc9_59.2) [symbolic = %.loc9_62 (constants.%.61f)]
// CHECK:STDOUT:   %B.impl_witness.loc9_62.2: <witness> = impl_witness %B.impl_witness_table, @BB.as_type.as.B.impl(%AA.loc9_16.2, %BB.loc9_26.2) [symbolic = %B.impl_witness.loc9_62.2 (constants.%B.impl_witness.089)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_60: <witness> = require_complete_type %B.type.loc9_60.2 [symbolic = %require_complete.loc9_60 (constants.%require_complete.58dcfe.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_49 as %B.type.loc9_60.1 {
// CHECK:STDOUT:     %B.impl_witness_table = impl_witness_table (), @BB.as_type.as.B.impl [concrete]
// CHECK:STDOUT:     %B.impl_witness.loc9_62.1: <witness> = impl_witness %B.impl_witness_table, @BB.as_type.as.B.impl(constants.%AA, constants.%BB.a11) [symbolic = %B.impl_witness.loc9_62.2 (constants.%B.impl_witness.089)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %B.type.loc9_60.1
// CHECK:STDOUT:     witness = %B.impl_witness.loc9_62.1
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
// CHECK:STDOUT:     .Self = constants.%D.743
// CHECK:STDOUT:     .G = %D.G.decl
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.G(@D.%DD.loc14_11.2: type, %T.loc15_17.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = symbolic_binding DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %ptr.loc21_14.2: type = ptr_type %DD [symbolic = %ptr.loc21_14.2 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %B.type.loc21_15.2: type = facet_type <@B, @B(%ptr.loc21_14.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.904f2f.2)]
// CHECK:STDOUT:   %.loc21_7.2: require_specific_def_type = require_specific_def @BB.as.B.impl(%ptr.loc21_14.2, %T.loc15_17.1) [symbolic = %.loc21_7.2 (constants.%.5dd)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_17.1, @B, @B(%ptr.loc21_14.2) [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.ab0)]
// CHECK:STDOUT:   %B.facet.loc21_7.2: @D.G.%B.type.loc21_15.2 (%B.type.904f2f.2) = facet_value %T.loc15_17.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.ed9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_17.2 [symbolic = %T.loc15_17.1 (constants.%T.091)]
// CHECK:STDOUT:     %B.ref: %B.type.d1b = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc14_11.2 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %ptr.loc21_14.1: type = ptr_type %DD.ref [symbolic = %ptr.loc21_14.2 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %B.type.loc21_15.1: type = facet_type <@B, @B(constants.%ptr.e8f8f9.2)> [symbolic = %B.type.loc21_15.2 (constants.%B.type.904f2f.2)]
// CHECK:STDOUT:     %B.facet.loc21_7.1: @D.G.%B.type.loc21_15.2 (%B.type.904f2f.2) = facet_value %T.ref, (constants.%B.lookup_impl_witness.ab0) [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.ed9)]
// CHECK:STDOUT:     %.loc21_7.1: @D.G.%B.type.loc21_15.2 (%B.type.904f2f.2) = converted %T.ref, %B.facet.loc21_7.1 [symbolic = %B.facet.loc21_7.2 (constants.%B.facet.ed9)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.as_type.as.B.impl(constants.%AA, constants.%BB.a11) {
// CHECK:STDOUT:   %AA.patt.loc9_16.2 => constants.%AA.patt
// CHECK:STDOUT:   %AA.loc9_16.2 => constants.%AA
// CHECK:STDOUT:   %A.type.loc9_32.2 => constants.%A.type.fa003b.1
// CHECK:STDOUT:   %.Self.frozen.loc9_34.2 => constants.%.Self.frozen.b15
// CHECK:STDOUT:   %require_complete.loc9_40 => constants.%require_complete.792
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type
// CHECK:STDOUT:   %assoc0 => constants.%assoc0
// CHECK:STDOUT:   %A.lookup_impl_witness.loc9_40.1 => constants.%A.lookup_impl_witness.030
// CHECK:STDOUT:   %impl.elem0.loc9_40.3 => constants.%impl.elem0.fba
// CHECK:STDOUT:   %.Self => constants.%.Self.8ed243.1
// CHECK:STDOUT:   %A.lookup_impl_witness.loc9_40.2 => constants.%A.lookup_impl_witness.9e772d.1
// CHECK:STDOUT:   %impl.elem0.loc9_40.4 => constants.%impl.elem0.7f79de.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.83dbed.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b5
// CHECK:STDOUT:   %BB.patt.loc9_26.2 => constants.%BB.patt.954
// CHECK:STDOUT:   %BB.loc9_26.2 => constants.%BB.a11
// CHECK:STDOUT:   %BB.as_type.loc9_49.2 => constants.%BB.as_type
// CHECK:STDOUT:   %ptr.loc9_59.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %B.type.loc9_60.2 => constants.%B.type.904f2f.1
// CHECK:STDOUT:   %.loc9_62 => constants.%.61f
// CHECK:STDOUT:   %B.impl_witness.loc9_62.2 => constants.%B.impl_witness.089
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.patt.loc14_11.2 => constants.%DD.patt
// CHECK:STDOUT:   %DD.loc14_11.1 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.cc0
// CHECK:STDOUT:   %D.G => constants.%D.G.6b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%DD, constants.%T.091) {
// CHECK:STDOUT:   %T.patt.loc15_17.2 => constants.%T.patt.cb8
// CHECK:STDOUT:   %T.loc15_17.1 => constants.%T.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.patt.loc14_11.2 => constants.%DD.patt
// CHECK:STDOUT:   %DD.loc14_11.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.G.type => constants.%D.G.type.bec
// CHECK:STDOUT:   %D.G => constants.%D.G.50b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.G(constants.%empty_struct_type, constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc15_17.2 => constants.%T.patt.cb8
// CHECK:STDOUT:   %T.loc15_17.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc21_14.2 => constants.%ptr.c28
// CHECK:STDOUT:   %B.type.loc21_15.2 => constants.%B.type.5e6
// CHECK:STDOUT:   %.loc21_7.2 => constants.%.e6e
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.295
// CHECK:STDOUT:   %B.facet.loc21_7.2 => constants.%B.facet.b2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
