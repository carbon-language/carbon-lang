// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE+7]]:5: error: cannot convert type `T` into type implementing `A(DD) where .(A(DD).X) = DD` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     C(DD).F(T);
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_dot_self_symbolic_type.carbon:[[@LINE-8]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! A(CC) where .X = CC) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    C(DD).F(T);
  }
}

fn H(T:! type) {
  D({}).G(T);
}

// --- dot_self_symbolic_in_impl_lookup.carbon

interface A(AA:! type) { let X:! type; }

interface B(AA:! type) {}

impl forall [AA:! type, BB:! A(AA) where .X = ()] BB as B(AA*) {}

impl forall [AA:! type, BB:! type] BB as B(AA) {}

class D(DD:! type) {
  fn G(T:! type) {
    // TODO: This produces a `lookup_impl_witness` instruction looking
    // for `.Self` as `A(DD)` in the generic eval block for `G`. That
    // is from an impl we ended up not using, so shouldn't be included
    // in the eval block.
    //
    // No SemIR dump ranges here: the lookup_impl_witness instruction
    // is hard to associate with a range.
    T as B(DD*);
  }
}

fn H(T:! type) {
  D({}).G(T);
}

// CHECK:STDOUT: --- fail_dot_self_symbolic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AA: type = bind_symbolic_name AA, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.75a876.1: type = facet_type <@A, @A(%AA)> [symbolic]
// CHECK:STDOUT:   %Self.75344f.1: %A.type.75a876.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.1: type = assoc_entity_type @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.1: %A.assoc_type.ed3b32.1 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e1b5.1: type = class_type @C, @C(%CC) [symbolic]
// CHECK:STDOUT:   %A.type.75a876.2: type = facet_type <@A, @A(%CC)> [symbolic]
// CHECK:STDOUT:   %.Self.fafe34.1: %A.type.75a876.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.75344f.2: %A.type.75a876.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.2: type = assoc_entity_type @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.2: %A.assoc_type.ed3b32.2 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %require_complete.6daa34.1: <witness> = require_complete_type %A.type.75a876.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1bbfba.1: type = facet_access_type %.Self.fafe34.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.4e49fb.1: <witness> = lookup_impl_witness %.Self.fafe34.1, @A, @A(%CC) [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type.75a876.2 = facet_value %.Self.as_type.1bbfba.1, (%A.lookup_impl_witness.4e49fb.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.452c56.1: type = impl_witness_access %A.lookup_impl_witness.4e49fb.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.5b404c.1: type = facet_type <@A, @A(%CC) where %impl.elem0.452c56.1 = %CC> [symbolic]
// CHECK:STDOUT:   %T.dad: %A_where.type.5b404c.1 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.2e945c.1: type = pattern_type %A_where.type.5b404c.1 [symbolic]
// CHECK:STDOUT:   %F.type.704ae6.1: type = fn_type @F, @C(%CC) [symbolic]
// CHECK:STDOUT:   %F.2d247e.1: %F.type.704ae6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.af4: type = class_type @D, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %G.type.0da: type = fn_type @G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %G.54c: %G.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.54c, @G(%empty_struct_type, %T.8b3) [symbolic]
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.loc4_13.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc4_13.2 (constants.%AA)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %CC.patt: %pattern_type.98f = symbolic_binding_pattern CC, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CC.loc6_9.1: type = bind_symbolic_name CC, 0 [symbolic = %CC.loc6_9.2 (constants.%CC)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %DD.patt: %pattern_type.98f = symbolic_binding_pattern DD, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DD.loc13_9.1: type = bind_symbolic_name DD, 0 [symbolic = %DD.loc13_9.2 (constants.%DD)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%AA.loc4_13.1: type) {
// CHECK:STDOUT:   %AA.loc4_13.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc4_13.2 (constants.%AA)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%AA.loc4_13.2)> [symbolic = %A.type (constants.%A.type.75a876.1)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a876.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.75344f.1)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc4_13.2) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.1)]
// CHECK:STDOUT:   %assoc0: @A.%A.assoc_type (%A.assoc_type.ed3b32.1) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a876.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.75344f.1)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @A.%A.assoc_type (%A.assoc_type.ed3b32.1) = assoc_entity element0, @A.%X [symbolic = @A.%assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@A.%AA.loc4_13.1: type, @A.%Self.1: @A.%A.type (%A.type.75a876.1)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%CC.loc6_9.1: type) {
// CHECK:STDOUT:   %CC.loc6_9.2: type = bind_symbolic_name CC, 0 [symbolic = %CC.loc6_9.2 (constants.%CC)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%CC.loc6_9.2) [symbolic = %F.type (constants.%F.type.704ae6.1)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type.704ae6.1) = struct_value () [symbolic = %F (constants.%F.2d247e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type.704ae6.1) = fn_decl @F [symbolic = @C.%F (constants.%F.2d247e.1)] {
// CHECK:STDOUT:       %T.patt: @F.%pattern_type (%pattern_type.2e945c.1) = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_18.1: type = splice_block %.loc10_18.2 [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)] {
// CHECK:STDOUT:         %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:         %CC.ref.loc10_14: type = name_ref CC, @C.%CC.loc6_9.1 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %A.type.loc10_16.2: type = facet_type <@A, @A(constants.%CC)> [symbolic = %A.type.loc10_16.1 (constants.%A.type.75a876.2)]
// CHECK:STDOUT:         %.Self.2: @F.%A.type.loc10_16.1 (%A.type.75a876.2) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:         %.Self.ref: @F.%A.type.loc10_16.1 (%A.type.75a876.2) = name_ref .Self, %.Self.2 [symbolic = %.Self.1 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:         %.loc10_24.1: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = specific_constant @X.%assoc0, @A(constants.%CC) [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %X.ref: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = name_ref X, %.loc10_24.1 [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:         %.Self.as_type.loc10_24.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc10_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:         %.loc10_24.2: type = converted %.Self.ref, %.Self.as_type.loc10_24.2 [symbolic = %.Self.as_type.loc10_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:         %impl.elem0.loc10_24.2: type = impl_witness_access constants.%A.lookup_impl_witness.4e49fb.1, element0 [symbolic = %impl.elem0.loc10_24.1 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:         %CC.ref.loc10_29: type = name_ref CC, @C.%CC.loc6_9.1 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:         %.loc10_18.2: type = where_expr %.Self.2 [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)] {
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc10_24.2, %CC.ref.loc10_29
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc10_8.2: @F.%A_where.type (%A_where.type.5b404c.1) = bind_symbolic_name T, 1 [symbolic = %T.loc10_8.1 (constants.%T.dad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic class @D(%DD.loc13_9.1: type) {
// CHECK:STDOUT:   %DD.loc13_9.2: type = bind_symbolic_name DD, 0 [symbolic = %DD.loc13_9.2 (constants.%DD)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @D(%DD.loc13_9.2) [symbolic = %G.type (constants.%G.type.a94)]
// CHECK:STDOUT:   %G: @D.%G.type (%G.type.a94) = struct_value () [symbolic = %G (constants.%G.714)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %G.decl: @D.%G.type (%G.type.a94) = fn_decl @G [symbolic = @D.%G (constants.%G.714)] {
// CHECK:STDOUT:       %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.loc14_8.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @F(@C.%CC.loc6_9.1: type, %T.loc10_8.2: @F.%A_where.type (%A_where.type.5b404c.1)) {
// CHECK:STDOUT:   %CC: type = bind_symbolic_name CC, 0 [symbolic = %CC (constants.%CC)]
// CHECK:STDOUT:   %A.type.loc10_16.1: type = facet_type <@A, @A(%CC)> [symbolic = %A.type.loc10_16.1 (constants.%A.type.75a876.2)]
// CHECK:STDOUT:   %.Self.1: @F.%A.type.loc10_16.1 (%A.type.75a876.2) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc10_16.1 [symbolic = %require_complete (constants.%require_complete.6daa34.1)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%CC) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.2)]
// CHECK:STDOUT:   %assoc0: @F.%A.assoc_type (%A.assoc_type.ed3b32.2) = assoc_entity element0, @A.%X [symbolic = %assoc0 (constants.%assoc0.ce3509.2)]
// CHECK:STDOUT:   %.Self.as_type.loc10_24.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc10_24.1 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @A, @A(%CC) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.4e49fb.1)]
// CHECK:STDOUT:   %impl.elem0.loc10_24.1: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_24.1 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%CC) where %impl.elem0.loc10_24.1 = %CC> [symbolic = %A_where.type (constants.%A_where.type.5b404c.1)]
// CHECK:STDOUT:   %T.loc10_8.1: @F.%A_where.type (%A_where.type.5b404c.1) = bind_symbolic_name T, 1 [symbolic = %T.loc10_8.1 (constants.%T.dad)]
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
// CHECK:STDOUT: generic fn @G(@D.%DD.loc13_9.1: type, %T.loc14_8.2: type) {
// CHECK:STDOUT:   %T.loc14_8.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %C.loc15_9.2: type = class_type @C, @C(%DD) [symbolic = %C.loc15_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc15_9.2 [symbolic = %require_complete (constants.%require_complete.389)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%DD) [symbolic = %F.type (constants.%F.type.704ae6.2)]
// CHECK:STDOUT:   %F: @G.%F.type (%F.type.704ae6.2) = struct_value () [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%DD)> [symbolic = %A.type (constants.%A.type.75a876.3)]
// CHECK:STDOUT:   %.Self: @G.%A.type (%A.type.75a876.3) = bind_symbolic_name .Self [symbolic = %.Self (constants.%.Self.fafe34.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @A, @A(%DD) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.4e49fb.2)]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.452c56.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc13_9.1 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %C.loc15_9.1: type = class_type @C, @C(constants.%DD) [symbolic = %C.loc15_9.2 (constants.%C.f2e1b5.2)]
// CHECK:STDOUT:     %.loc15: @G.%F.type (%F.type.704ae6.2) = specific_constant @C.%F.decl, @C(constants.%DD) [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:     %F.ref: @G.%F.type (%F.type.704ae6.2) = name_ref F, %.loc15 [symbolic = %F (constants.%F.2d247e.2)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_8.2 [symbolic = %T.loc14_8.1 (constants.%T.336)]
// CHECK:STDOUT:     %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%DD, <error>) [concrete = <error>]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn() [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.specific_fn.loc20_8.2: <specific function> = specific_function constants.%G.54c, @G(constants.%empty_struct_type, %T.loc19_6.2) [symbolic = %G.specific_fn.loc20_8.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %.loc20_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_7: type = converted %.loc20_6, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%empty_struct_type) [concrete = constants.%D.af4]
// CHECK:STDOUT:     %.loc20_8: %G.type.0da = specific_constant @D.%G.decl, @D(constants.%empty_struct_type) [concrete = constants.%G.54c]
// CHECK:STDOUT:     %G.ref: %G.type.0da = name_ref G, %.loc20_8 [concrete = constants.%G.54c]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %G.specific_fn.loc20_8.1: <specific function> = specific_function %G.ref, @G(constants.%empty_struct_type, constants.%T.8b3) [symbolic = %G.specific_fn.loc20_8.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc20_8.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%AA) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%AA
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%AA, constants.%Self.75344f.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%CC) {
// CHECK:STDOUT:   %CC.loc6_9.2 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.704ae6.1
// CHECK:STDOUT:   %F => constants.%F.2d247e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%CC) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%CC
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.75a876.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.75344f.2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%CC, constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%CC, constants.%T.dad) {
// CHECK:STDOUT:   %CC => constants.%CC
// CHECK:STDOUT:   %A.type.loc10_16.1 => constants.%A.type.75a876.2
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.fafe34.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6daa34.1
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.2
// CHECK:STDOUT:   %.Self.as_type.loc10_24.1 => constants.%.Self.as_type.1bbfba.1
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.1
// CHECK:STDOUT:   %impl.elem0.loc10_24.1 => constants.%impl.elem0.452c56.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.5b404c.1
// CHECK:STDOUT:   %T.loc10_8.1 => constants.%T.dad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2e945c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc13_9.2 => constants.%DD
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
// CHECK:STDOUT: specific @A(constants.%DD) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%DD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%DD, <error>) {
// CHECK:STDOUT:   %CC => constants.%DD
// CHECK:STDOUT:   %A.type.loc10_16.1 => constants.%A.type.75a876.3
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.fafe34.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6daa34.2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.3
// CHECK:STDOUT:   %.Self.as_type.loc10_24.1 => constants.%.Self.as_type.1bbfba.2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.2
// CHECK:STDOUT:   %impl.elem0.loc10_24.1 => constants.%impl.elem0.452c56.2
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.5b404c.2
// CHECK:STDOUT:   %T.loc10_8.1 => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2e945c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc13_9.2 => constants.%empty_struct_type
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
// CHECK:STDOUT:   %C.loc15_9.2 => constants.%C.7a7
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
// CHECK:STDOUT: specific @A(constants.%empty_struct_type) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%empty_struct_type
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
// CHECK:STDOUT:   %A.type.75a876.1: type = facet_type <@A, @A(%AA)> [symbolic]
// CHECK:STDOUT:   %Self.753: %A.type.75a876.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.1: type = assoc_entity_type @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.1: %A.assoc_type.ed3b32.1 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %B.type.bf2: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.bf2 = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.243: type = facet_type <@B, @B(%AA)> [symbolic]
// CHECK:STDOUT:   %Self.e06: %B.type.243 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.Self.fafe34.1: %A.type.75a876.1 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete.6daa34.1: <witness> = require_complete_type %A.type.75a876.1 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1bbfba.1: type = facet_access_type %.Self.fafe34.1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.4e49fb.1: <witness> = lookup_impl_witness %.Self.fafe34.1, @A, @A(%AA) [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type.75a876.1 = facet_value %.Self.as_type.1bbfba.1, (%A.lookup_impl_witness.4e49fb.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.452c56.1: type = impl_witness_access %A.lookup_impl_witness.4e49fb.1, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.812090.1: type = facet_type <@A, @A(%AA) where %impl.elem0.452c56.1 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %BB.a66: %A_where.type.812090.1 = bind_symbolic_name BB, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.0b2895.1: type = pattern_type %A_where.type.812090.1 [symbolic]
// CHECK:STDOUT:   %BB.as_type: type = facet_access_type %BB.a66 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %AA [symbolic]
// CHECK:STDOUT:   %B.type.4e3fc4.1: type = facet_type <@B, @B(%ptr.79f131.1)> [symbolic]
// CHECK:STDOUT:   %Self.beb: %B.type.4e3fc4.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.13ba89.1: <witness> = require_complete_type %B.type.4e3fc4.1 [symbolic]
// CHECK:STDOUT:   %B.impl_witness.741: <witness> = impl_witness file.%B.impl_witness_table.loc6, @impl.e5b(%AA, %BB.a66) [symbolic]
// CHECK:STDOUT:   %BB.336: type = bind_symbolic_name BB, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.19b: <witness> = require_complete_type %B.type.243 [symbolic]
// CHECK:STDOUT:   %B.impl_witness.500: <witness> = impl_witness file.%B.impl_witness_table.loc8, @impl.929(%AA, %BB.336) [symbolic]
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.432: type = class_type @D, @D(%DD) [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %G.type.a94: type = fn_type @G, @D(%DD) [symbolic]
// CHECK:STDOUT:   %G.714: %G.type.a94 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %DD [symbolic]
// CHECK:STDOUT:   %B.type.4e3fc4.2: type = facet_type <@B, @B(%ptr.79f131.2)> [symbolic]
// CHECK:STDOUT:   %A.type.75a876.2: type = facet_type <@A, @A(%DD)> [symbolic]
// CHECK:STDOUT:   %.Self.fafe34.2: %A.type.75a876.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.4e49fb.2: <witness> = lookup_impl_witness %.Self.fafe34.2, @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %impl.elem0.452c56.2: type = impl_witness_access %A.lookup_impl_witness.4e49fb.2, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.812090.2: type = facet_type <@A, @A(%DD) where %impl.elem0.452c56.2 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.6daa34.2: <witness> = require_complete_type %A.type.75a876.2 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3b32.2: type = assoc_entity_type @A, @A(%DD) [symbolic]
// CHECK:STDOUT:   %assoc0.ce3509.2: %A.assoc_type.ed3b32.2 = assoc_entity element0, @A.%X [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1bbfba.2: type = facet_access_type %.Self.fafe34.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.0b2895.2: type = pattern_type %A_where.type.812090.2 [symbolic]
// CHECK:STDOUT:   %require_complete.13ba89.2: <witness> = require_complete_type %B.type.4e3fc4.2 [symbolic]
// CHECK:STDOUT:   %B.impl_witness.9d1: <witness> = impl_witness file.%B.impl_witness_table.loc8, @impl.929(%ptr.79f131.2, %T.336) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.8e3: <witness> = lookup_impl_witness %T.336, @B, @B(%ptr.79f131.2) [symbolic]
// CHECK:STDOUT:   %B.facet.44e: %B.type.4e3fc4.2 = facet_value %T.336, (%B.lookup_impl_witness.8e3) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.af4: type = class_type @D, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %G.type.0da: type = fn_type @G, @D(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %G.54c: %G.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.54c, @G(%empty_struct_type, %T.8b3) [symbolic]
// CHECK:STDOUT:   %B.type.973: type = facet_type <@B, @B(%ptr.c28)> [concrete]
// CHECK:STDOUT:   %A.type.700: type = facet_type <@A, @A(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.Self.4e2: %A.type.700 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.935: <witness> = lookup_impl_witness %.Self.4e2, @A, @A(%empty_struct_type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ab3: type = impl_witness_access %A.lookup_impl_witness.935, element0 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.cf4: type = facet_type <@A, @A(%empty_struct_type) where %impl.elem0.ab3 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Self.20c: %A.type.700 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.assoc_type.fb9: type = assoc_entity_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.1e3: %A.assoc_type.fb9 = assoc_entity element0, @A.%X [concrete]
// CHECK:STDOUT:   %complete_type.393: <witness> = complete_type_witness %A.type.700 [concrete]
// CHECK:STDOUT:   %.Self.as_type.ac1: type = facet_access_type %.Self.4e2 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a26: type = pattern_type %A_where.type.cf4 [concrete]
// CHECK:STDOUT:   %Self.2af: %B.type.973 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.bb7: <witness> = complete_type_witness %B.type.973 [concrete]
// CHECK:STDOUT:   %B.impl_witness.1a2: <witness> = impl_witness file.%B.impl_witness_table.loc8, @impl.929(%ptr.c28, %T.8b3) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.c4b: <witness> = lookup_impl_witness %T.8b3, @B, @B(%ptr.c28) [symbolic]
// CHECK:STDOUT:   %B.facet.8ed: %B.type.973 = facet_value %T.8b3, (%B.lookup_impl_witness.c4b) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.loc2_13.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc2_13.2 (constants.%AA)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type.bf2 = interface_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.loc4_13.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc4_13.2 (constants.%AA)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.e5b [concrete] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %BB.patt: @impl.e5b.%pattern_type (%pattern_type.0b2895.1) = symbolic_binding_pattern BB, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %BB.ref: @impl.e5b.%A_where.type (%A_where.type.812090.1) = name_ref BB, %BB.loc6_25.1 [symbolic = %BB.loc6_25.2 (constants.%BB.a66)]
// CHECK:STDOUT:     %BB.as_type.loc6_51.1: type = facet_access_type %BB.ref [symbolic = %BB.as_type.loc6_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %.loc6_51: type = converted %BB.ref, %BB.as_type.loc6_51.1 [symbolic = %BB.as_type.loc6_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:     %B.ref: %B.type.bf2 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %AA.ref.loc6_59: type = name_ref AA, %AA.loc6_14.1 [symbolic = %AA.loc6_14.2 (constants.%AA)]
// CHECK:STDOUT:     %ptr.loc6_61.1: type = ptr_type %AA.ref.loc6_59 [symbolic = %ptr.loc6_61.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %B.type.loc6_62.1: type = facet_type <@B, @B(constants.%ptr.79f131.1)> [symbolic = %B.type.loc6_62.2 (constants.%B.type.4e3fc4.1)]
// CHECK:STDOUT:     %AA.loc6_14.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_14.2 (constants.%AA)]
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.2 [symbolic = %A_where.type (constants.%A_where.type.812090.1)] {
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %AA.ref.loc6_32: type = name_ref AA, %AA.loc6_14.1 [symbolic = %AA.loc6_14.2 (constants.%AA)]
// CHECK:STDOUT:       %A.type.loc6_34.1: type = facet_type <@A, @A(constants.%AA)> [symbolic = %A.type.loc6_34.2 (constants.%A.type.75a876.1)]
// CHECK:STDOUT:       %.Self.1: @impl.e5b.%A.type.loc6_34.2 (%A.type.75a876.1) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:       %.Self.ref: @impl.e5b.%A.type.loc6_34.2 (%A.type.75a876.1) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:       %.loc6_42.1: @impl.e5b.%A.assoc_type (%A.assoc_type.ed3b32.1) = specific_constant @X.%assoc0, @A(constants.%AA) [symbolic = %assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:       %X.ref: @impl.e5b.%A.assoc_type (%A.assoc_type.ed3b32.1) = name_ref X, %.loc6_42.1 [symbolic = %assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:       %.Self.as_type.loc6_42.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc6_42.2 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:       %.loc6_42.2: type = converted %.Self.ref, %.Self.as_type.loc6_42.1 [symbolic = %.Self.as_type.loc6_42.2 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:       %impl.elem0.loc6_42.1: type = impl_witness_access constants.%A.lookup_impl_witness.4e49fb.1, element0 [symbolic = %impl.elem0.loc6_42.2 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:       %.loc6_48.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_48.2: type = converted %.loc6_48.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc6_36.2: type = where_expr %.Self.1 [symbolic = %A_where.type (constants.%A_where.type.812090.1)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc6_42.1, %.loc6_48.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %BB.loc6_25.1: @impl.e5b.%A_where.type (%A_where.type.812090.1) = bind_symbolic_name BB, 1 [symbolic = %BB.loc6_25.2 (constants.%BB.a66)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table.loc6 = impl_witness_table (), @impl.e5b [concrete]
// CHECK:STDOUT:   %B.impl_witness.loc6: <witness> = impl_witness %B.impl_witness_table.loc6, @impl.e5b(constants.%AA, constants.%BB.a66) [symbolic = @impl.e5b.%B.impl_witness (constants.%B.impl_witness.741)]
// CHECK:STDOUT:   impl_decl @impl.929 [concrete] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.98f = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %BB.patt: %pattern_type.98f = symbolic_binding_pattern BB, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %BB.ref: type = name_ref BB, %BB.loc8_25.1 [symbolic = %BB.loc8_25.2 (constants.%BB.336)]
// CHECK:STDOUT:     %B.ref: %B.type.bf2 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %AA.ref: type = name_ref AA, %AA.loc8_14.1 [symbolic = %AA.loc8_14.2 (constants.%AA)]
// CHECK:STDOUT:     %B.type.loc8_46.1: type = facet_type <@B, @B(constants.%AA)> [symbolic = %B.type.loc8_46.2 (constants.%B.type.243)]
// CHECK:STDOUT:     %AA.loc8_14.1: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc8_14.2 (constants.%AA)]
// CHECK:STDOUT:     %BB.loc8_25.1: type = bind_symbolic_name BB, 1 [symbolic = %BB.loc8_25.2 (constants.%BB.336)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table.loc8 = impl_witness_table (), @impl.929 [concrete]
// CHECK:STDOUT:   %B.impl_witness.loc8: <witness> = impl_witness %B.impl_witness_table.loc8, @impl.929(constants.%AA, constants.%BB.336) [symbolic = @impl.929.%B.impl_witness (constants.%B.impl_witness.500)]
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %DD.patt: %pattern_type.98f = symbolic_binding_pattern DD, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DD.loc10_9.1: type = bind_symbolic_name DD, 0 [symbolic = %DD.loc10_9.2 (constants.%DD)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc16_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%AA.loc2_13.1: type) {
// CHECK:STDOUT:   %AA.loc2_13.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc2_13.2 (constants.%AA)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%AA.loc2_13.2)> [symbolic = %A.type (constants.%A.type.75a876.1)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.75a876.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc2_13.2) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.1)]
// CHECK:STDOUT:   %assoc0: @A.%A.assoc_type (%A.assoc_type.ed3b32.1) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.75a876.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.753)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @A.%A.assoc_type (%A.assoc_type.ed3b32.1) = assoc_entity element0, @A.%X [symbolic = @A.%assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @B(%AA.loc4_13.1: type) {
// CHECK:STDOUT:   %AA.loc4_13.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc4_13.2 (constants.%AA)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = facet_type <@B, @B(%AA.loc4_13.2)> [symbolic = %B.type (constants.%B.type.243)]
// CHECK:STDOUT:   %Self.2: @B.%B.type (%B.type.243) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @B.%B.type (%B.type.243) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.e06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@A.%AA.loc2_13.1: type, @A.%Self.1: @A.%A.type (%A.type.75a876.1)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.e5b(%AA.loc6_14.1: type, %BB.loc6_25.1: @impl.e5b.%A_where.type (%A_where.type.812090.1)) {
// CHECK:STDOUT:   %AA.loc6_14.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_14.2 (constants.%AA)]
// CHECK:STDOUT:   %A.type.loc6_34.2: type = facet_type <@A, @A(%AA.loc6_14.2)> [symbolic = %A.type.loc6_34.2 (constants.%A.type.75a876.1)]
// CHECK:STDOUT:   %.Self.2: @impl.e5b.%A.type.loc6_34.2 (%A.type.75a876.1) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.fafe34.1)]
// CHECK:STDOUT:   %require_complete.loc6_42: <witness> = require_complete_type %A.type.loc6_34.2 [symbolic = %require_complete.loc6_42 (constants.%require_complete.6daa34.1)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%AA.loc6_14.2) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3b32.1)]
// CHECK:STDOUT:   %assoc0: @impl.e5b.%A.assoc_type (%A.assoc_type.ed3b32.1) = assoc_entity element0, @A.%X [symbolic = %assoc0 (constants.%assoc0.ce3509.1)]
// CHECK:STDOUT:   %.Self.as_type.loc6_42.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc6_42.2 (constants.%.Self.as_type.1bbfba.1)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @A, @A(%AA.loc6_14.2) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.4e49fb.1)]
// CHECK:STDOUT:   %impl.elem0.loc6_42.2: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_42.2 (constants.%impl.elem0.452c56.1)]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A, @A(%AA.loc6_14.2) where %impl.elem0.loc6_42.2 = constants.%empty_tuple.type> [symbolic = %A_where.type (constants.%A_where.type.812090.1)]
// CHECK:STDOUT:   %BB.loc6_25.2: @impl.e5b.%A_where.type (%A_where.type.812090.1) = bind_symbolic_name BB, 1 [symbolic = %BB.loc6_25.2 (constants.%BB.a66)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type [symbolic = %pattern_type (constants.%pattern_type.0b2895.1)]
// CHECK:STDOUT:   %BB.as_type.loc6_51.2: type = facet_access_type %BB.loc6_25.2 [symbolic = %BB.as_type.loc6_51.2 (constants.%BB.as_type)]
// CHECK:STDOUT:   %ptr.loc6_61.2: type = ptr_type %AA.loc6_14.2 [symbolic = %ptr.loc6_61.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %B.type.loc6_62.2: type = facet_type <@B, @B(%ptr.loc6_61.2)> [symbolic = %B.type.loc6_62.2 (constants.%B.type.4e3fc4.1)]
// CHECK:STDOUT:   %require_complete.loc6_62: <witness> = require_complete_type %B.type.loc6_62.2 [symbolic = %require_complete.loc6_62 (constants.%require_complete.13ba89.1)]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness file.%B.impl_witness_table.loc6, @impl.e5b(%AA.loc6_14.2, %BB.loc6_25.2) [symbolic = %B.impl_witness (constants.%B.impl_witness.741)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc6_51 as %B.type.loc6_62.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%B.impl_witness.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.929(%AA.loc8_14.1: type, %BB.loc8_25.1: type) {
// CHECK:STDOUT:   %AA.loc8_14.2: type = bind_symbolic_name AA, 0 [symbolic = %AA.loc8_14.2 (constants.%AA)]
// CHECK:STDOUT:   %BB.loc8_25.2: type = bind_symbolic_name BB, 1 [symbolic = %BB.loc8_25.2 (constants.%BB.336)]
// CHECK:STDOUT:   %B.type.loc8_46.2: type = facet_type <@B, @B(%AA.loc8_14.2)> [symbolic = %B.type.loc8_46.2 (constants.%B.type.243)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %B.type.loc8_46.2 [symbolic = %require_complete (constants.%require_complete.19b)]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness file.%B.impl_witness_table.loc8, @impl.929(%AA.loc8_14.2, %BB.loc8_25.2) [symbolic = %B.impl_witness (constants.%B.impl_witness.500)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %BB.ref as %B.type.loc8_46.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%B.impl_witness.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%DD.loc10_9.1: type) {
// CHECK:STDOUT:   %DD.loc10_9.2: type = bind_symbolic_name DD, 0 [symbolic = %DD.loc10_9.2 (constants.%DD)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @D(%DD.loc10_9.2) [symbolic = %G.type (constants.%G.type.a94)]
// CHECK:STDOUT:   %G: @D.%G.type (%G.type.a94) = struct_value () [symbolic = %G (constants.%G.714)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %G.decl: @D.%G.type (%G.type.a94) = fn_decl @G [symbolic = @D.%G (constants.%G.714)] {
// CHECK:STDOUT:       %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.loc11_8.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc11_8.1 (constants.%T.336)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.432
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .DD = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@D.%DD.loc10_9.1: type, %T.loc11_8.2: type) {
// CHECK:STDOUT:   %T.loc11_8.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc11_8.1 (constants.%T.336)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD: type = bind_symbolic_name DD, 0 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:   %ptr.loc12_14.2: type = ptr_type %DD [symbolic = %ptr.loc12_14.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:   %B.type.loc12_15.2: type = facet_type <@B, @B(%ptr.loc12_14.2)> [symbolic = %B.type.loc12_15.2 (constants.%B.type.4e3fc4.2)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%DD)> [symbolic = %A.type (constants.%A.type.75a876.2)]
// CHECK:STDOUT:   %.Self: @G.%A.type (%A.type.75a876.2) = bind_symbolic_name .Self [symbolic = %.Self (constants.%.Self.fafe34.2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @A, @A(%DD) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.4e49fb.2)]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.452c56.2)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_8.1, @B, @B(%ptr.loc12_14.2) [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.8e3)]
// CHECK:STDOUT:   %B.facet.loc12_7.2: @G.%B.type.loc12_15.2 (%B.type.4e3fc4.2) = facet_value %T.loc11_8.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc12_7.2 (constants.%B.facet.44e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_8.2 [symbolic = %T.loc11_8.1 (constants.%T.336)]
// CHECK:STDOUT:     %B.ref: %B.type.bf2 = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %DD.ref: type = name_ref DD, @D.%DD.loc10_9.1 [symbolic = %DD (constants.%DD)]
// CHECK:STDOUT:     %ptr.loc12_14.1: type = ptr_type %DD.ref [symbolic = %ptr.loc12_14.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:     %B.type.loc12_15.1: type = facet_type <@B, @B(constants.%ptr.79f131.2)> [symbolic = %B.type.loc12_15.2 (constants.%B.type.4e3fc4.2)]
// CHECK:STDOUT:     %B.facet.loc12_7.1: @G.%B.type.loc12_15.2 (%B.type.4e3fc4.2) = facet_value constants.%T.336, (constants.%B.lookup_impl_witness.8e3) [symbolic = %B.facet.loc12_7.2 (constants.%B.facet.44e)]
// CHECK:STDOUT:     %.loc12: @G.%B.type.loc12_15.2 (%B.type.4e3fc4.2) = converted %T.ref, %B.facet.loc12_7.1 [symbolic = %B.facet.loc12_7.2 (constants.%B.facet.44e)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc16_6.1: type) {
// CHECK:STDOUT:   %T.loc16_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.specific_fn.loc17_8.2: <specific function> = specific_function constants.%G.54c, @G(constants.%empty_struct_type, %T.loc16_6.2) [symbolic = %G.specific_fn.loc17_8.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %.loc17_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_7: type = converted %.loc17_6, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%empty_struct_type) [concrete = constants.%D.af4]
// CHECK:STDOUT:     %.loc17_8: %G.type.0da = specific_constant @D.%G.decl, @D(constants.%empty_struct_type) [concrete = constants.%G.54c]
// CHECK:STDOUT:     %G.ref: %G.type.0da = name_ref G, %.loc17_8 [concrete = constants.%G.54c]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_6.1 [symbolic = %T.loc16_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %G.specific_fn.loc17_8.1: <specific function> = specific_function %G.ref, @G(constants.%empty_struct_type, constants.%T.8b3) [symbolic = %G.specific_fn.loc17_8.2 (constants.%G.specific_fn)]
// CHECK:STDOUT:     %G.call: init %empty_tuple.type = call %G.specific_fn.loc17_8.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%AA) {
// CHECK:STDOUT:   %AA.loc2_13.2 => constants.%AA
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.75a876.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.753
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%AA, constants.%Self.753) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%AA) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%AA
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type.243
// CHECK:STDOUT:   %Self.2 => constants.%Self.e06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%AA, constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%ptr.79f131.1) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type.4e3fc4.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.beb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.e5b(constants.%AA, constants.%BB.a66) {
// CHECK:STDOUT:   %AA.loc6_14.2 => constants.%AA
// CHECK:STDOUT:   %A.type.loc6_34.2 => constants.%A.type.75a876.1
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.fafe34.1
// CHECK:STDOUT:   %require_complete.loc6_42 => constants.%require_complete.6daa34.1
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.1
// CHECK:STDOUT:   %.Self.as_type.loc6_42.2 => constants.%.Self.as_type.1bbfba.1
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.1
// CHECK:STDOUT:   %impl.elem0.loc6_42.2 => constants.%impl.elem0.452c56.1
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.812090.1
// CHECK:STDOUT:   %BB.loc6_25.2 => constants.%BB.a66
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0b2895.1
// CHECK:STDOUT:   %BB.as_type.loc6_51.2 => constants.%BB.as_type
// CHECK:STDOUT:   %ptr.loc6_61.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %B.type.loc6_62.2 => constants.%B.type.4e3fc4.1
// CHECK:STDOUT:   %require_complete.loc6_62 => constants.%require_complete.13ba89.1
// CHECK:STDOUT:   %B.impl_witness => constants.%B.impl_witness.741
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.929(constants.%AA, constants.%BB.336) {
// CHECK:STDOUT:   %AA.loc8_14.2 => constants.%AA
// CHECK:STDOUT:   %BB.loc8_25.2 => constants.%BB.336
// CHECK:STDOUT:   %B.type.loc8_46.2 => constants.%B.type.243
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19b
// CHECK:STDOUT:   %B.impl_witness => constants.%B.impl_witness.500
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%DD) {
// CHECK:STDOUT:   %DD.loc10_9.2 => constants.%DD
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.a94
// CHECK:STDOUT:   %G => constants.%G.714
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%DD, constants.%T.336) {
// CHECK:STDOUT:   %T.loc11_8.1 => constants.%T.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%ptr.79f131.2) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%ptr.79f131.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%DD) {
// CHECK:STDOUT:   %AA.loc2_13.2 => constants.%DD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.e5b(constants.%DD, <error>) {
// CHECK:STDOUT:   %AA.loc6_14.2 => constants.%DD
// CHECK:STDOUT:   %A.type.loc6_34.2 => constants.%A.type.75a876.2
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.fafe34.2
// CHECK:STDOUT:   %require_complete.loc6_42 => constants.%require_complete.6daa34.2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.ed3b32.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce3509.2
// CHECK:STDOUT:   %.Self.as_type.loc6_42.2 => constants.%.Self.as_type.1bbfba.2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.4e49fb.2
// CHECK:STDOUT:   %impl.elem0.loc6_42.2 => constants.%impl.elem0.452c56.2
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.812090.2
// CHECK:STDOUT:   %BB.loc6_25.2 => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0b2895.2
// CHECK:STDOUT:   %BB.as_type.loc6_51.2 => <error>
// CHECK:STDOUT:   %ptr.loc6_61.2 => constants.%ptr.79f131.2
// CHECK:STDOUT:   %B.type.loc6_62.2 => constants.%B.type.4e3fc4.2
// CHECK:STDOUT:   %require_complete.loc6_62 => constants.%require_complete.13ba89.2
// CHECK:STDOUT:   %B.impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.929(constants.%ptr.79f131.2, constants.%T.336) {
// CHECK:STDOUT:   %AA.loc8_14.2 => constants.%ptr.79f131.2
// CHECK:STDOUT:   %BB.loc8_25.2 => constants.%T.336
// CHECK:STDOUT:   %B.type.loc8_46.2 => constants.%B.type.4e3fc4.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.13ba89.2
// CHECK:STDOUT:   %B.impl_witness => constants.%B.impl_witness.9d1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc16_6.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%empty_struct_type) {
// CHECK:STDOUT:   %DD.loc10_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.0da
// CHECK:STDOUT:   %G => constants.%G.54c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_struct_type, constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc11_8.1 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DD => constants.%empty_struct_type
// CHECK:STDOUT:   %ptr.loc12_14.2 => constants.%ptr.c28
// CHECK:STDOUT:   %B.type.loc12_15.2 => constants.%B.type.973
// CHECK:STDOUT:   %A.type => constants.%A.type.700
// CHECK:STDOUT:   %.Self => constants.%.Self.4e2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.935
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.ab3
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.c4b
// CHECK:STDOUT:   %B.facet.loc12_7.2 => constants.%B.facet.8ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%ptr.c28) {
// CHECK:STDOUT:   %AA.loc4_13.2 => constants.%ptr.c28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type.973
// CHECK:STDOUT:   %Self.2 => constants.%Self.2af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%empty_struct_type) {
// CHECK:STDOUT:   %AA.loc2_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.700
// CHECK:STDOUT:   %Self.2 => constants.%Self.20c
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.fb9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.e5b(constants.%empty_struct_type, <error>) {
// CHECK:STDOUT:   %AA.loc6_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.type.loc6_34.2 => constants.%A.type.700
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.4e2
// CHECK:STDOUT:   %require_complete.loc6_42 => constants.%complete_type.393
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.fb9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e3
// CHECK:STDOUT:   %.Self.as_type.loc6_42.2 => constants.%.Self.as_type.ac1
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.935
// CHECK:STDOUT:   %impl.elem0.loc6_42.2 => constants.%impl.elem0.ab3
// CHECK:STDOUT:   %A_where.type => constants.%A_where.type.cf4
// CHECK:STDOUT:   %BB.loc6_25.2 => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a26
// CHECK:STDOUT:   %BB.as_type.loc6_51.2 => <error>
// CHECK:STDOUT:   %ptr.loc6_61.2 => constants.%ptr.c28
// CHECK:STDOUT:   %B.type.loc6_62.2 => constants.%B.type.973
// CHECK:STDOUT:   %require_complete.loc6_62 => constants.%complete_type.bb7
// CHECK:STDOUT:   %B.impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.929(constants.%ptr.c28, constants.%T.8b3) {
// CHECK:STDOUT:   %AA.loc8_14.2 => constants.%ptr.c28
// CHECK:STDOUT:   %BB.loc8_25.2 => constants.%T.8b3
// CHECK:STDOUT:   %B.type.loc8_46.2 => constants.%B.type.973
// CHECK:STDOUT:   %require_complete => constants.%complete_type.bb7
// CHECK:STDOUT:   %B.impl_witness => constants.%B.impl_witness.1a2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
