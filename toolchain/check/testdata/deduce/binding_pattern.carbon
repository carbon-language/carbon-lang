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
  fn Create(value: T) {}
}

fn F(U:! type, V:! type) {
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE+10]]:15: error: cannot implicitly convert expression of type `{}` to `V` [ConversionFailure]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE+7]]:15: note: type `{}` does not implement interface `Core.ImplicitAs(V)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_incompatible_deduce.carbon:[[@LINE-10]]:13: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Create(value: T) {}
  // CHECK:STDERR:             ^~~~~~~~
  // CHECK:STDERR:
  C(V).Create({});
}

// --- fail_todo_compatible_deduce.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  fn Create(value: T) {}
}

// TODO: This `where` should be sufficient to say that `{} as V` works.
fn F(U:! type, V:! type where {} impls Core.ImplicitAs(.Self)) {
  // CHECK:STDERR: fail_todo_compatible_deduce.carbon:[[@LINE+10]]:15: error: cannot implicitly convert expression of type `{}` to `V` [ConversionFailure]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_todo_compatible_deduce.carbon:[[@LINE+7]]:15: note: type `{}` does not implement interface `Core.ImplicitAs(V)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   C(V).Create({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_todo_compatible_deduce.carbon:[[@LINE-11]]:13: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Create(value: T) {}
  // CHECK:STDERR:             ^~~~~~~~
  // CHECK:STDERR:
  C(V).Create({});
}

// CHECK:STDOUT: --- fail_incompatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.c11: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.723: %C.Create.type.c11 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.ee9: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.type.c3d: type = fn_type @C.Create, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.b40: %C.Create.type.c3d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.1fc: <witness> = require_complete_type %C.ee9 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %V [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.b40, @C.Create(%V) [symbolic]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.485: type = facet_type <@ImplicitAs, @ImplicitAs(%V)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.d88: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V) [symbolic]
// CHECK:STDOUT:   %assoc0.2aa: %ImplicitAs.assoc_type.d88 = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.74e: <witness> = require_complete_type %ImplicitAs.type.485 [symbolic]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.import_ref.218: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc8_6.2: type = symbolic_binding U, 0 [symbolic = %U.loc8_6.1 (constants.%U)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.loc8_16.2: type = symbolic_binding V, 1 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.c11)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.c11) = struct_value () [symbolic = %C.Create (constants.%C.Create.723)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.c11) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.723)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
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
// CHECK:STDOUT: generic fn @F(%U.loc8_6.2: type, %V.loc8_16.2: type) {
// CHECK:STDOUT:   %U.loc8_6.1: type = symbolic_binding U, 0 [symbolic = %U.loc8_6.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc8_16.1: type = symbolic_binding V, 1 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc19_6.2: type = class_type @C, @C(%V.loc8_16.1) [symbolic = %C.loc19_6.2 (constants.%C.ee9)]
// CHECK:STDOUT:   %require_complete.loc19_7: <witness> = require_complete_type %C.loc19_6.2 [symbolic = %require_complete.loc19_7 (constants.%require_complete.1fc)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.loc8_16.1) [symbolic = %C.Create.type (constants.%C.Create.type.c3d)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.c3d) = struct_value () [symbolic = %C.Create (constants.%C.Create.b40)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc19_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.loc8_16.1) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc19_16.1: <witness> = require_complete_type %V.loc8_16.1 [symbolic = %require_complete.loc19_16.1 (constants.%require_complete.441)]
// CHECK:STDOUT:   %ImplicitAs.type.loc19_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.loc8_16.1)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.485)]
// CHECK:STDOUT:   %require_complete.loc19_16.2: <witness> = require_complete_type %ImplicitAs.type.loc19_16.2 [symbolic = %require_complete.loc19_16.2 (constants.%require_complete.74e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.loc8_16.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.d88)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d88) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0 (constants.%assoc0.2aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc8_16.2 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:     %C.loc19_6.1: type = class_type @C, @C(constants.%V) [symbolic = %C.loc19_6.2 (constants.%C.ee9)]
// CHECK:STDOUT:     %.loc19_7: @F.%C.Create.type (%C.Create.type.c3d) = specific_constant @C.%C.Create.decl, @C(constants.%V) [symbolic = %C.Create (constants.%C.Create.b40)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.c3d) = name_ref Create, %.loc19_7 [symbolic = %C.Create (constants.%C.Create.b40)]
// CHECK:STDOUT:     %.loc19_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.Create.specific_fn.loc19_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc19_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.485)]
// CHECK:STDOUT:     %.loc19_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d88) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%V) [symbolic = %assoc0 (constants.%assoc0.2aa)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d88) = name_ref Convert, %.loc19_16.2 [symbolic = %assoc0 (constants.%assoc0.2aa)]
// CHECK:STDOUT:     %.loc19_16.3: @F.%V.loc8_16.1 (%V) = converted %.loc19_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.Create.call: init %empty_tuple.type = call %C.Create.specific_fn.loc19_7.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.c11
// CHECK:STDOUT:   %C.Create => constants.%C.Create.723
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc8_6.1 => constants.%U
// CHECK:STDOUT:   %V.loc8_16.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.c3d
// CHECK:STDOUT:   %C.Create => constants.%C.Create.b40
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
// CHECK:STDOUT: --- fail_todo_compatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.c11: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.723: %C.Create.type.c11 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.16f: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c54: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.16f)> [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %V: %type_where = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b07: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 1, %V [symbolic]
// CHECK:STDOUT:   %C.267: type = class_type @C, @C(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.type.0e1: type = fn_type @C.Create, @C(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.0d5: %C.Create.type.0e1 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.496: <witness> = require_complete_type %C.267 [symbolic]
// CHECK:STDOUT:   %pattern_type.3a1: type = pattern_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.0d5, @C.Create(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.073: <witness> = require_complete_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3bd: type = facet_type <@ImplicitAs, @ImplicitAs(%V.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.3af: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.b14: %ImplicitAs.assoc_type.3af = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.723: <witness> = require_complete_type %ImplicitAs.type.3bd [symbolic]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.import_ref.218: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.b07 = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %U.loc9_6.2: type = symbolic_binding U, 0 [symbolic = %U.loc9_6.1 (constants.%U)]
// CHECK:STDOUT:     %.loc9_25.1: type = splice_block %.loc9_25.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:       %.Self.3: type = symbolic_binding .Self [symbolic_self = constants.%.Self.16f]
// CHECK:STDOUT:       %.loc9_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.16f]
// CHECK:STDOUT:       %ImplicitAs.type.loc9: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.16f)> [symbolic_self = constants.%ImplicitAs.type.c54]
// CHECK:STDOUT:       %.loc9_32.2: type = converted %.loc9_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_25.2: type = where_expr %.Self.3 [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_base_facet_type type
// CHECK:STDOUT:         requirement_impls %.loc9_32.2, %ImplicitAs.type.loc9
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_16.2: %type_where = symbolic_binding V, 1 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.c11)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.c11) = struct_value () [symbolic = %C.Create (constants.%C.Create.723)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.c11) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.723)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
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
// CHECK:STDOUT: generic fn @F(%U.loc9_6.2: type, %V.loc9_16.2: %type_where) {
// CHECK:STDOUT:   %U.loc9_6.1: type = symbolic_binding U, 0 [symbolic = %U.loc9_6.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc9_16.1: %type_where = symbolic_binding V, 1 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 1, %V.loc9_16.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %C.loc20_6.2: type = class_type @C, @C(%V.binding.as_type) [symbolic = %C.loc20_6.2 (constants.%C.267)]
// CHECK:STDOUT:   %require_complete.loc20_7: <witness> = require_complete_type %C.loc20_6.2 [symbolic = %require_complete.loc20_7 (constants.%require_complete.496)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.binding.as_type) [symbolic = %C.Create.type (constants.%C.Create.type.0e1)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.0e1) = struct_value () [symbolic = %C.Create (constants.%C.Create.0d5)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc20_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.binding.as_type) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc20_16.1: <witness> = require_complete_type %V.binding.as_type [symbolic = %require_complete.loc20_16.1 (constants.%require_complete.073)]
// CHECK:STDOUT:   %ImplicitAs.type.loc20_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.binding.as_type)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.3bd)]
// CHECK:STDOUT:   %require_complete.loc20_16.2: <witness> = require_complete_type %ImplicitAs.type.loc20_16.2 [symbolic = %require_complete.loc20_16.2 (constants.%require_complete.723)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.binding.as_type) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.3af)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.3af) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0 (constants.%assoc0.b14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: %type_where = name_ref V, %V.loc9_16.2 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type: type = facet_access_type %V.ref [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %.loc20_6: type = converted %V.ref, %V.as_type [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %C.loc20_6.1: type = class_type @C, @C(constants.%V.binding.as_type) [symbolic = %C.loc20_6.2 (constants.%C.267)]
// CHECK:STDOUT:     %.loc20_7: @F.%C.Create.type (%C.Create.type.0e1) = specific_constant @C.%C.Create.decl, @C(constants.%V.binding.as_type) [symbolic = %C.Create (constants.%C.Create.0d5)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.0e1) = name_ref Create, %.loc20_7 [symbolic = %C.Create (constants.%C.Create.0d5)]
// CHECK:STDOUT:     %.loc20_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.Create.specific_fn.loc20_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V.binding.as_type) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc20_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V.binding.as_type)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.3bd)]
// CHECK:STDOUT:     %.loc20_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.3af) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%V.binding.as_type) [symbolic = %assoc0 (constants.%assoc0.b14)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.3af) = name_ref Convert, %.loc20_16.2 [symbolic = %assoc0 (constants.%assoc0.b14)]
// CHECK:STDOUT:     %.loc20_16.3: @F.%V.binding.as_type (%V.binding.as_type) = converted %.loc20_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.Create.call: init %empty_tuple.type = call %C.Create.specific_fn.loc20_7.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.c11
// CHECK:STDOUT:   %C.Create => constants.%C.Create.723
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc9_6.1 => constants.%U
// CHECK:STDOUT:   %V.loc9_16.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V.binding.as_type) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%V.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.0e1
// CHECK:STDOUT:   %C.Create => constants.%C.Create.0d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V.binding.as_type) {
// CHECK:STDOUT:   %T => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3a1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.073
// CHECK:STDOUT: }
// CHECK:STDOUT:
