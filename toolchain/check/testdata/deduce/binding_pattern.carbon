// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/binding_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/binding_pattern.carbon

// --- fail_incompatible_deduce.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  fn Create(unused value: T) {}
}

fn F(unused U:! type, V:! type) {
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE+10]]:15: error: cannot implicitly convert expression of type `{}` to `V` [ConversionFailure]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE+7]]:15: note: type `{}` does not implement interface `Core.ImplicitAs(V)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE-10]]:20: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Create(unused value: T) {}
  // CHECK:STDERR:                    ^~~~~~~~
  // CHECK:STDERR:
  C(V).Create({});
}

// --- compatible_deduce.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  fn Create(unused value: T) {}
}

// This `where` is sufficient to say that `{} as V` works.
fn F(unused U:! type, V:! Core.Destroy where {} impls Core.ImplicitAs(.Self)) {
  C(V).Create({});
}

// CHECK:STDOUT: --- fail_incompatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.346: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.9a4: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.ece: %C.Create.type.9a4 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c61: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.type.9ea: type = fn_type @C.Create, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.afc: %C.Create.type.9ea = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.171: <witness> = require_complete_type %C.c61 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %V [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.afc, @C.Create(%V) [symbolic]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.152: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.f66: %ImplicitAs.type.152 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.6dd: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f65: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.f66) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.020: %ImplicitAs.WithSelf.Convert.type.f65 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d61: type = facet_type <@ImplicitAs, @ImplicitAs(%V)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.25d: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V) [symbolic]
// CHECK:STDOUT:   %assoc0.d31: %ImplicitAs.assoc_type.25d = assoc_entity element0, imports.%Core.import_ref.b56 [symbolic]
// CHECK:STDOUT:   %require_complete.cf2: <witness> = require_complete_type %ImplicitAs.type.d61 [symbolic]
// CHECK:STDOUT:   %assoc0.a16: %ImplicitAs.assoc_type.6dd = assoc_entity element0, imports.%Core.import_ref.b1d [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f8a: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.6dd) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.a16)]
// CHECK:STDOUT:   %Core.import_ref.b56: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.f65) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.020)]
// CHECK:STDOUT:   %Core.import_ref.b1d = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: type = symbolic_binding U, 0 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_27.1: type = splice_block %.loc8_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_24.2: type = symbolic_binding V, 1 [symbolic = %V.loc8_24.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_10.1) [symbolic = %C.Create.type (constants.%C.Create.type.9a4)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.9a4) = struct_value () [symbolic = %C.Create (constants.%C.Create.ece)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.9a4) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.ece)] {
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = at_binding_pattern value, %value.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_10.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.346
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%value.param: @C.Create.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%U.loc8_14.2: type, %V.loc8_24.2: type) {
// CHECK:STDOUT:   %U.loc8_14.1: type = symbolic_binding U, 0 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc8_24.1: type = symbolic_binding V, 1 [symbolic = %V.loc8_24.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc19_6.2: type = class_type @C, @C(%V.loc8_24.1) [symbolic = %C.loc19_6.2 (constants.%C.c61)]
// CHECK:STDOUT:   %require_complete.loc19_7: <witness> = require_complete_type %C.loc19_6.2 [symbolic = %require_complete.loc19_7 (constants.%require_complete.171)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.loc8_24.1) [symbolic = %C.Create.type (constants.%C.Create.type.9ea)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.9ea) = struct_value () [symbolic = %C.Create (constants.%C.Create.afc)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc19_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.loc8_24.1) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc19_16.1: <witness> = require_complete_type %V.loc8_24.1 [symbolic = %require_complete.loc19_16.1 (constants.%require_complete.441)]
// CHECK:STDOUT:   %ImplicitAs.type.loc19_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.loc8_24.1)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.d61)]
// CHECK:STDOUT:   %require_complete.loc19_16.2: <witness> = require_complete_type %ImplicitAs.type.loc19_16.2 [symbolic = %require_complete.loc19_16.2 (constants.%require_complete.cf2)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.loc8_24.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.25d)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.25d) = assoc_entity element0, imports.%Core.import_ref.b56 [symbolic = %assoc0 (constants.%assoc0.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc8_24.2 [symbolic = %V.loc8_24.1 (constants.%V)]
// CHECK:STDOUT:     %C.loc19_6.1: type = class_type @C, @C(constants.%V) [symbolic = %C.loc19_6.2 (constants.%C.c61)]
// CHECK:STDOUT:     %.loc19_7: @F.%C.Create.type (%C.Create.type.9ea) = specific_constant @C.%C.Create.decl, @C(constants.%V) [symbolic = %C.Create (constants.%C.Create.afc)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.9ea) = name_ref Create, %.loc19_7 [symbolic = %C.Create (constants.%C.Create.afc)]
// CHECK:STDOUT:     %.loc19_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.Create.specific_fn.loc19_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc19_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.d61)]
// CHECK:STDOUT:     %.loc19_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.25d) = specific_constant imports.%Core.import_ref.f8a, @ImplicitAs.WithSelf(constants.%V, constants.%Self.f66) [symbolic = %assoc0 (constants.%assoc0.d31)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.25d) = name_ref Convert, %.loc19_16.2 [symbolic = %assoc0 (constants.%assoc0.d31)]
// CHECK:STDOUT:     %.loc19_16.3: @F.%V.loc8_24.1 (%V) = converted %.loc19_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.Create.call: init %empty_tuple.type = call %C.Create.specific_fn.loc19_7.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.9a4
// CHECK:STDOUT:   %C.Create => constants.%C.Create.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.loc8_24.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.9ea
// CHECK:STDOUT:   %C.Create => constants.%C.Create.afc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.946
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.441
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.346: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.9a4: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.ece: %C.Create.type.9a4 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %.Self.312: %Destroy.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.152: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.f66: %ImplicitAs.type.152 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.6dd: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f65: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.f66) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.020: %ImplicitAs.WithSelf.Convert.type.f65 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.312 [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.527: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.as_type)> [symbolic_self]
// CHECK:STDOUT:   %Destroy_where.type: type = facet_type <@Destroy where %empty_struct_type impls @ImplicitAs, @ImplicitAs(%.Self.as_type)> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.20b: type = pattern_type %Destroy_where.type [symbolic_self]
// CHECK:STDOUT:   %V: %Destroy_where.type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %C.8a7: type = class_type @C, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.type.5ac: type = fn_type @C.Create, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.a77: %C.Create.type.5ac = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.6a4: <witness> = require_complete_type %C.8a7 [symbolic]
// CHECK:STDOUT:   %pattern_type.c6d: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.a77, @C.Create(%V.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.b23: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.799: type = facet_type <@ImplicitAs, @ImplicitAs(%V.as_type)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.092: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.3ae: %ImplicitAs.assoc_type.092 = assoc_entity element0, imports.%Core.import_ref.b56 [symbolic]
// CHECK:STDOUT:   %require_complete.100: <witness> = require_complete_type %ImplicitAs.type.799 [symbolic]
// CHECK:STDOUT:   %assoc0.a16: %ImplicitAs.assoc_type.6dd = assoc_entity element0, imports.%Core.import_ref.b1d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %V, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %V.as_type, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.lookup_impl_witness: <witness> = lookup_impl_witness %empty_struct_type, @ImplicitAs, @ImplicitAs(%V.as_type) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.799 = facet_value %empty_struct_type, (%ImplicitAs.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f42: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%V.as_type, %ImplicitAs.facet) [symbolic]
// CHECK:STDOUT:   %.e5f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f42, %ImplicitAs.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.d88: %.e5f = impl_witness_access %ImplicitAs.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.a8d: <bound method> = bound_method %empty_struct, %impl.elem0.d88 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7bd: <specific function> = specific_impl_function %impl.elem0.d88, @ImplicitAs.WithSelf.Convert(%V.as_type, %ImplicitAs.facet) [symbolic]
// CHECK:STDOUT:   %bound_method.54e: <bound method> = bound_method %empty_struct, %specific_impl_fn.7bd [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a0c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.073: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a0c, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.69b: %.073 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e82: <specific function> = specific_impl_function %impl.elem0.69b, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f8a: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.6dd) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.a16)]
// CHECK:STDOUT:   %Core.import_ref.b56: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.f65) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.020)]
// CHECK:STDOUT:   %Core.import_ref.b1d = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.20b = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc9_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_14.2: type = symbolic_binding U, 0 [symbolic = %U.loc9_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc9_40.1: type = splice_block %.loc9_40.2 [symbolic_self = constants.%Destroy_where.type] {
// CHECK:STDOUT:       %.Self.loc9_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Core.ref.loc9_27: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %.Self.loc9_40: %Destroy.type = symbolic_binding .Self [symbolic_self = constants.%.Self.312]
// CHECK:STDOUT:       %.loc9_47.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %Core.ref.loc9_55: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref: %Destroy.type = name_ref .Self, %.Self.loc9_40 [symbolic_self = constants.%.Self.312]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_76: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %ImplicitAs.type.loc9: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.as_type)> [symbolic_self = constants.%ImplicitAs.type.527]
// CHECK:STDOUT:       %.loc9_47.2: type = converted %.loc9_47.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_40.2: type = where_expr [symbolic_self = constants.%Destroy_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type %Destroy.ref
// CHECK:STDOUT:         requirement_impls %.loc9_47.2, %ImplicitAs.type.loc9
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_24.2: %Destroy_where.type = symbolic_binding V, 1 [symbolic = %V.loc9_24.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_10.1) [symbolic = %C.Create.type (constants.%C.Create.type.9a4)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.9a4) = struct_value () [symbolic = %C.Create (constants.%C.Create.ece)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.9a4) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.ece)] {
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = at_binding_pattern value, %value.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_10.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.346
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%value.param: @C.Create.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%U.loc9_14.2: type, %V.loc9_24.2: %Destroy_where.type) {
// CHECK:STDOUT:   %U.loc9_14.1: type = symbolic_binding U, 0 [symbolic = %U.loc9_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc9_24.1: %Destroy_where.type = symbolic_binding V, 1 [symbolic = %V.loc9_24.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.as_type.loc10_6.2: type = facet_access_type %V.loc9_24.1 [symbolic = %V.as_type.loc10_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %C.loc10_6.2: type = class_type @C, @C(%V.as_type.loc10_6.2) [symbolic = %C.loc10_6.2 (constants.%C.8a7)]
// CHECK:STDOUT:   %require_complete.loc10_7: <witness> = require_complete_type %C.loc10_6.2 [symbolic = %require_complete.loc10_7 (constants.%require_complete.6a4)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.as_type.loc10_6.2) [symbolic = %C.Create.type (constants.%C.Create.type.5ac)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.5ac) = struct_value () [symbolic = %C.Create (constants.%C.Create.a77)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc10_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.as_type.loc10_6.2) [symbolic = %C.Create.specific_fn.loc10_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc10_16.1: <witness> = require_complete_type %V.as_type.loc10_6.2 [symbolic = %require_complete.loc10_16.1 (constants.%require_complete.b23)]
// CHECK:STDOUT:   %ImplicitAs.type.loc10_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.as_type.loc10_6.2)> [symbolic = %ImplicitAs.type.loc10_16.2 (constants.%ImplicitAs.type.799)]
// CHECK:STDOUT:   %require_complete.loc10_16.2: <witness> = require_complete_type %ImplicitAs.type.loc10_16.2 [symbolic = %require_complete.loc10_16.2 (constants.%require_complete.100)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.as_type.loc10_6.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.092)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.092) = assoc_entity element0, imports.%Core.import_ref.b56 [symbolic = %assoc0 (constants.%assoc0.3ae)]
// CHECK:STDOUT:   %ImplicitAs.lookup_impl_witness: <witness> = lookup_impl_witness constants.%empty_struct_type, @ImplicitAs, @ImplicitAs(%V.as_type.loc10_6.2) [symbolic = %ImplicitAs.lookup_impl_witness (constants.%ImplicitAs.lookup_impl_witness)]
// CHECK:STDOUT:   %ImplicitAs.facet: @F.%ImplicitAs.type.loc10_16.2 (%ImplicitAs.type.799) = facet_value constants.%empty_struct_type, (%ImplicitAs.lookup_impl_witness) [symbolic = %ImplicitAs.facet (constants.%ImplicitAs.facet)]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%V.as_type.loc10_6.2, %ImplicitAs.facet) [symbolic = %ImplicitAs.WithSelf.Convert.type (constants.%ImplicitAs.WithSelf.Convert.type.f42)]
// CHECK:STDOUT:   %.loc10_16.8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type, %ImplicitAs.facet [symbolic = %.loc10_16.8 (constants.%.e5f)]
// CHECK:STDOUT:   %impl.elem0.loc10_16.3: @F.%.loc10_16.8 (%.e5f) = impl_witness_access %ImplicitAs.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_16.3 (constants.%impl.elem0.d88)]
// CHECK:STDOUT:   %bound_method.loc10_16.5: <bound method> = bound_method constants.%empty_struct, %impl.elem0.loc10_16.3 [symbolic = %bound_method.loc10_16.5 (constants.%bound_method.a8d)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_16.3: <specific function> = specific_impl_function %impl.elem0.loc10_16.3, @ImplicitAs.WithSelf.Convert(%V.as_type.loc10_6.2, %ImplicitAs.facet) [symbolic = %specific_impl_fn.loc10_16.3 (constants.%specific_impl_fn.7bd)]
// CHECK:STDOUT:   %bound_method.loc10_16.6: <bound method> = bound_method constants.%empty_struct, %specific_impl_fn.loc10_16.3 [symbolic = %bound_method.loc10_16.6 (constants.%bound_method.54e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc9_24.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %V.as_type.loc10_6.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.a0c)]
// CHECK:STDOUT:   %.loc10_16.9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc10_16.9 (constants.%.073)]
// CHECK:STDOUT:   %impl.elem0.loc10_16.4: @F.%.loc10_16.9 (%.073) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_16.4 (constants.%impl.elem0.69b)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_16.4: <specific function> = specific_impl_function %impl.elem0.loc10_16.4, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc10_16.4 (constants.%specific_impl_fn.e82)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: %Destroy_where.type = name_ref V, %V.loc9_24.2 [symbolic = %V.loc9_24.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc10_6.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc10_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc10_6: type = converted %V.ref, %V.as_type.loc10_6.1 [symbolic = %V.as_type.loc10_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %C.loc10_6.1: type = class_type @C, @C(constants.%V.as_type) [symbolic = %C.loc10_6.2 (constants.%C.8a7)]
// CHECK:STDOUT:     %.loc10_7: @F.%C.Create.type (%C.Create.type.5ac) = specific_constant @C.%C.Create.decl, @C(constants.%V.as_type) [symbolic = %C.Create (constants.%C.Create.a77)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.5ac) = name_ref Create, %.loc10_7 [symbolic = %C.Create (constants.%C.Create.a77)]
// CHECK:STDOUT:     %.loc10_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.Create.specific_fn.loc10_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V.as_type) [symbolic = %C.Create.specific_fn.loc10_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc10_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V.as_type)> [symbolic = %ImplicitAs.type.loc10_16.2 (constants.%ImplicitAs.type.799)]
// CHECK:STDOUT:     %.loc10_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.092) = specific_constant imports.%Core.import_ref.f8a, @ImplicitAs.WithSelf(constants.%V.as_type, constants.%Self.f66) [symbolic = %assoc0 (constants.%assoc0.3ae)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.092) = name_ref Convert, %.loc10_16.2 [symbolic = %assoc0 (constants.%assoc0.3ae)]
// CHECK:STDOUT:     %impl.elem0.loc10_16.1: @F.%.loc10_16.8 (%.e5f) = impl_witness_access constants.%ImplicitAs.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_16.3 (constants.%impl.elem0.d88)]
// CHECK:STDOUT:     %bound_method.loc10_16.1: <bound method> = bound_method %.loc10_16.1, %impl.elem0.loc10_16.1 [symbolic = %bound_method.loc10_16.5 (constants.%bound_method.a8d)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_16.1: <specific function> = specific_impl_function %impl.elem0.loc10_16.1, @ImplicitAs.WithSelf.Convert(constants.%V.as_type, constants.%ImplicitAs.facet) [symbolic = %specific_impl_fn.loc10_16.3 (constants.%specific_impl_fn.7bd)]
// CHECK:STDOUT:     %bound_method.loc10_16.2: <bound method> = bound_method %.loc10_16.1, %specific_impl_fn.loc10_16.1 [symbolic = %bound_method.loc10_16.6 (constants.%bound_method.54e)]
// CHECK:STDOUT:     %.loc10_16.3: ref @F.%V.as_type.loc10_6.2 (%V.as_type) = temporary_storage
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_16.4: %empty_struct_type = converted %.loc10_16.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.call: init @F.%V.as_type.loc10_6.2 (%V.as_type) to %.loc10_16.3 = call %bound_method.loc10_16.2(%.loc10_16.4)
// CHECK:STDOUT:     %.loc10_16.5: init @F.%V.as_type.loc10_6.2 (%V.as_type) = converted %.loc10_16.1, %ImplicitAs.WithSelf.Convert.call
// CHECK:STDOUT:     %.loc10_16.6: ref @F.%V.as_type.loc10_6.2 (%V.as_type) = temporary %.loc10_16.3, %.loc10_16.5
// CHECK:STDOUT:     %.loc10_16.7: @F.%V.as_type.loc10_6.2 (%V.as_type) = acquire_value %.loc10_16.6
// CHECK:STDOUT:     %C.Create.call: init %empty_tuple.type = call %C.Create.specific_fn.loc10_7.1(%.loc10_16.7)
// CHECK:STDOUT:     %impl.elem0.loc10_16.2: @F.%.loc10_16.9 (%.073) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_16.4 (constants.%impl.elem0.69b)]
// CHECK:STDOUT:     %bound_method.loc10_16.3: <bound method> = bound_method %.loc10_16.6, %impl.elem0.loc10_16.2
// CHECK:STDOUT:     %specific_impl_fn.loc10_16.2: <specific function> = specific_impl_function %impl.elem0.loc10_16.2, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc10_16.4 (constants.%specific_impl_fn.e82)]
// CHECK:STDOUT:     %bound_method.loc10_16.4: <bound method> = bound_method %.loc10_16.6, %specific_impl_fn.loc10_16.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc10_16.4(%.loc10_16.6)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.9a4
// CHECK:STDOUT:   %C.Create => constants.%C.Create.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc9_14.1 => constants.%U
// CHECK:STDOUT:   %V.loc9_24.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V.as_type) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%V.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.5ac
// CHECK:STDOUT:   %C.Create => constants.%C.Create.a77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V.as_type) {
// CHECK:STDOUT:   %T => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c6d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b23
// CHECK:STDOUT: }
// CHECK:STDOUT:
