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
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e6836e.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.2a3: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.135: %C.Create.type.2a3 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f62: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.type.c39: type = fn_type @C.Create, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.6a8: %C.Create.type.c39 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.a90: <witness> = require_complete_type %C.f62 [symbolic]
// CHECK:STDOUT:   %pattern_type.984: type = pattern_type %V [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.6a8, @C.Create(%V) [symbolic]
// CHECK:STDOUT:   %require_complete.162: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c8: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.e9f: %ImplicitAs.Convert.type.4c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b5: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.af3: type = facet_type <@ImplicitAs, @ImplicitAs(%V)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8fb: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V) [symbolic]
// CHECK:STDOUT:   %assoc0.123: %ImplicitAs.assoc_type.8fb = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic]
// CHECK:STDOUT:   %require_complete.78d: <witness> = require_complete_type %ImplicitAs.type.af3 [symbolic]
// CHECK:STDOUT:   %assoc0.7b6: %ImplicitAs.assoc_type.8b5 = assoc_entity element0, imports.%Core.import_ref.d11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b9c: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.7b6)]
// CHECK:STDOUT:   %Core.import_ref.b91: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %Core.import_ref.d11 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.2a3)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.2a3) = struct_value () [symbolic = %C.Create (constants.%C.Create.135)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.2a3) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.135)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.e6836e.1) = value_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.e6836e.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.e6836e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4b7)]
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
// CHECK:STDOUT:   %C.loc19_6.2: type = class_type @C, @C(%V.loc8_16.1) [symbolic = %C.loc19_6.2 (constants.%C.f62)]
// CHECK:STDOUT:   %require_complete.loc19_7: <witness> = require_complete_type %C.loc19_6.2 [symbolic = %require_complete.loc19_7 (constants.%require_complete.a90)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.loc8_16.1) [symbolic = %C.Create.type (constants.%C.Create.type.c39)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.c39) = struct_value () [symbolic = %C.Create (constants.%C.Create.6a8)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc19_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.loc8_16.1) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc19_16.1: <witness> = require_complete_type %V.loc8_16.1 [symbolic = %require_complete.loc19_16.1 (constants.%require_complete.162)]
// CHECK:STDOUT:   %ImplicitAs.type.loc19_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.loc8_16.1)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.af3)]
// CHECK:STDOUT:   %require_complete.loc19_16.2: <witness> = require_complete_type %ImplicitAs.type.loc19_16.2 [symbolic = %require_complete.loc19_16.2 (constants.%require_complete.78d)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.loc8_16.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.8fb)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8fb) = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic = %assoc0 (constants.%assoc0.123)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc8_16.2 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:     %C.loc19_6.1: type = class_type @C, @C(constants.%V) [symbolic = %C.loc19_6.2 (constants.%C.f62)]
// CHECK:STDOUT:     %.loc19_7: @F.%C.Create.type (%C.Create.type.c39) = specific_constant @C.%C.Create.decl, @C(constants.%V) [symbolic = %C.Create (constants.%C.Create.6a8)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.c39) = name_ref Create, %.loc19_7 [symbolic = %C.Create (constants.%C.Create.6a8)]
// CHECK:STDOUT:     %.loc19_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %C.Create.specific_fn.loc19_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc19_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.af3)]
// CHECK:STDOUT:     %.loc19_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8fb) = specific_constant imports.%Core.import_ref.b9c, @ImplicitAs(constants.%V) [symbolic = %assoc0 (constants.%assoc0.123)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8fb) = name_ref Convert, %.loc19_16.2 [symbolic = %assoc0 (constants.%assoc0.123)]
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.2a3
// CHECK:STDOUT:   %C.Create => constants.%C.Create.135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e6836e.1
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.c39
// CHECK:STDOUT:   %C.Create => constants.%C.Create.6a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.984
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_compatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e6836e.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.2a3: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.135: %C.Create.type.2a3 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.bcc: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c8: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.e9f: %ImplicitAs.Convert.type.4c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b5: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ca8: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.bcc)> [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %V: %type_where = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b07: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 1, %V [symbolic]
// CHECK:STDOUT:   %C.a6f: type = class_type @C, @C(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.type.715: type = fn_type @C.Create, @C(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.9da: %C.Create.type.715 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.071: <witness> = require_complete_type %C.a6f [symbolic]
// CHECK:STDOUT:   %pattern_type.dbe: type = pattern_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.9da, @C.Create(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.c4b: <witness> = require_complete_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2a9: type = facet_type <@ImplicitAs, @ImplicitAs(%V.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.975: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.binding.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.5ef: %ImplicitAs.assoc_type.975 = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic]
// CHECK:STDOUT:   %require_complete.800: <witness> = require_complete_type %ImplicitAs.type.2a9 [symbolic]
// CHECK:STDOUT:   %assoc0.7b6: %ImplicitAs.assoc_type.8b5 = assoc_entity element0, imports.%Core.import_ref.d11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b9c: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.7b6)]
// CHECK:STDOUT:   %Core.import_ref.b91: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %Core.import_ref.d11 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.b07 = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %U.loc9_6.2: type = symbolic_binding U, 0 [symbolic = %U.loc9_6.1 (constants.%U)]
// CHECK:STDOUT:     %.loc9_25.1: type = splice_block %.loc9_25.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %.Self.3: type = symbolic_binding .Self [symbolic_self = constants.%.Self.bcc]
// CHECK:STDOUT:       %.loc9_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.bcc]
// CHECK:STDOUT:       %ImplicitAs.type.loc9: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.bcc)> [symbolic_self = constants.%ImplicitAs.type.ca8]
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
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.2a3)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.2a3) = struct_value () [symbolic = %C.Create (constants.%C.Create.135)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.2a3) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.135)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.e6836e.1) = value_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.e6836e.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = value_binding value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.e6836e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4b7)]
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
// CHECK:STDOUT:   %C.loc20_6.2: type = class_type @C, @C(%V.binding.as_type) [symbolic = %C.loc20_6.2 (constants.%C.a6f)]
// CHECK:STDOUT:   %require_complete.loc20_7: <witness> = require_complete_type %C.loc20_6.2 [symbolic = %require_complete.loc20_7 (constants.%require_complete.071)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.binding.as_type) [symbolic = %C.Create.type (constants.%C.Create.type.715)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.715) = struct_value () [symbolic = %C.Create (constants.%C.Create.9da)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc20_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.binding.as_type) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc20_16.1: <witness> = require_complete_type %V.binding.as_type [symbolic = %require_complete.loc20_16.1 (constants.%require_complete.c4b)]
// CHECK:STDOUT:   %ImplicitAs.type.loc20_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.binding.as_type)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.2a9)]
// CHECK:STDOUT:   %require_complete.loc20_16.2: <witness> = require_complete_type %ImplicitAs.type.loc20_16.2 [symbolic = %require_complete.loc20_16.2 (constants.%require_complete.800)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.binding.as_type) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.975)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.975) = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic = %assoc0 (constants.%assoc0.5ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: %type_where = name_ref V, %V.loc9_16.2 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type: type = facet_access_type %V.ref [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %.loc20_6: type = converted %V.ref, %V.as_type [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %C.loc20_6.1: type = class_type @C, @C(constants.%V.binding.as_type) [symbolic = %C.loc20_6.2 (constants.%C.a6f)]
// CHECK:STDOUT:     %.loc20_7: @F.%C.Create.type (%C.Create.type.715) = specific_constant @C.%C.Create.decl, @C(constants.%V.binding.as_type) [symbolic = %C.Create (constants.%C.Create.9da)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.715) = name_ref Create, %.loc20_7 [symbolic = %C.Create (constants.%C.Create.9da)]
// CHECK:STDOUT:     %.loc20_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %C.Create.specific_fn.loc20_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V.binding.as_type) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc20_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V.binding.as_type)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.2a9)]
// CHECK:STDOUT:     %.loc20_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.975) = specific_constant imports.%Core.import_ref.b9c, @ImplicitAs(constants.%V.binding.as_type) [symbolic = %assoc0 (constants.%assoc0.5ef)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.975) = name_ref Convert, %.loc20_16.2 [symbolic = %assoc0 (constants.%assoc0.5ef)]
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.2a3
// CHECK:STDOUT:   %C.Create => constants.%C.Create.135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e6836e.1
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.715
// CHECK:STDOUT:   %C.Create => constants.%C.Create.9da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V.binding.as_type) {
// CHECK:STDOUT:   %T => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
