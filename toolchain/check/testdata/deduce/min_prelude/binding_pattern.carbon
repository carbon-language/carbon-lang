// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/min_prelude/binding_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/min_prelude/binding_pattern.carbon

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

//@dump-sem-ir-begin
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
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_compatible_deduce.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Create.type.f31: type = fn_type @Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %Create.cc8: %Create.type.f31 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.aba: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self)> [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %V: %type_where = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.344: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %C.cd6: type = class_type @C, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %Create.type.ff1: type = fn_type @Create, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %Create.96b: %Create.type.ff1 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.558: <witness> = require_complete_type %C.cd6 [symbolic]
// CHECK:STDOUT:   %pattern_type.6e0: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Create.specific_fn: <specific function> = specific_function %Create.96b, @Create(%V.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.0b5: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.113: type = facet_type <@ImplicitAs, @ImplicitAs(%V.as_type)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.83a: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.5dc: %ImplicitAs.assoc_type.83a = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.9fe: <witness> = require_complete_type %ImplicitAs.type.113 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude, loc14_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.344 = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc10_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc10_6.2 (constants.%U)]
// CHECK:STDOUT:     %.loc10_25.1: type = splice_block %.loc10_25.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc10_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %ImplicitAs.type.loc10: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self)> [symbolic_self = constants.%ImplicitAs.type.aba]
// CHECK:STDOUT:       %.loc10_32.2: type = converted %.loc10_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc10_25.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.loc10_32.2, %ImplicitAs.type.loc10
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc10_16.1: %type_where = bind_symbolic_name V, 1 [symbolic = %V.loc10_16.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_9.1: type) {
// CHECK:STDOUT:   %T.loc5_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Create.type: type = fn_type @Create, @C(%T.loc5_9.2) [symbolic = %Create.type (constants.%Create.type.f31)]
// CHECK:STDOUT:   %Create: @C.%Create.type (%Create.type.f31) = struct_value () [symbolic = %Create (constants.%Create.cc8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Create.decl: @C.%Create.type (%Create.type.f31) = fn_decl @Create [symbolic = @C.%Create (constants.%Create.cc8)] {
// CHECK:STDOUT:       %value.patt: @Create.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @Create.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc5_9.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @Create.%T (%T) = bind_name value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Create(@C.%T.loc5_9.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%value.param: @Create.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%U.loc10_6.1: type, %V.loc10_16.1: %type_where) {
// CHECK:STDOUT:   %U.loc10_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc10_6.2 (constants.%U)]
// CHECK:STDOUT:   %V.loc10_16.2: %type_where = bind_symbolic_name V, 1 [symbolic = %V.loc10_16.2 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.as_type.loc21_6.2: type = facet_access_type %V.loc10_16.2 [symbolic = %V.as_type.loc21_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %C.loc21_6.2: type = class_type @C, @C(%V.as_type.loc21_6.2) [symbolic = %C.loc21_6.2 (constants.%C.cd6)]
// CHECK:STDOUT:   %require_complete.loc21_7: <witness> = require_complete_type %C.loc21_6.2 [symbolic = %require_complete.loc21_7 (constants.%require_complete.558)]
// CHECK:STDOUT:   %Create.type: type = fn_type @Create, @C(%V.as_type.loc21_6.2) [symbolic = %Create.type (constants.%Create.type.ff1)]
// CHECK:STDOUT:   %Create: @F.%Create.type (%Create.type.ff1) = struct_value () [symbolic = %Create (constants.%Create.96b)]
// CHECK:STDOUT:   %Create.specific_fn.loc21_7.2: <specific function> = specific_function %Create, @Create(%V.as_type.loc21_6.2) [symbolic = %Create.specific_fn.loc21_7.2 (constants.%Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc21_16.1: <witness> = require_complete_type %V.as_type.loc21_6.2 [symbolic = %require_complete.loc21_16.1 (constants.%require_complete.0b5)]
// CHECK:STDOUT:   %ImplicitAs.type.loc21_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.as_type.loc21_6.2)> [symbolic = %ImplicitAs.type.loc21_16.2 (constants.%ImplicitAs.type.113)]
// CHECK:STDOUT:   %require_complete.loc21_16.2: <witness> = require_complete_type %ImplicitAs.type.loc21_16.2 [symbolic = %require_complete.loc21_16.2 (constants.%require_complete.9fe)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.as_type.loc21_6.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.83a)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: %type_where = name_ref V, %V.loc10_16.1 [symbolic = %V.loc10_16.2 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc21_6.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc21_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc21_6: type = converted %V.ref, %V.as_type.loc21_6.1 [symbolic = %V.as_type.loc21_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %C.loc21_6.1: type = class_type @C, @C(constants.%V.as_type) [symbolic = %C.loc21_6.2 (constants.%C.cd6)]
// CHECK:STDOUT:     %.loc21_7: @F.%Create.type (%Create.type.ff1) = specific_constant @C.%Create.decl, @C(constants.%V.as_type) [symbolic = %Create (constants.%Create.96b)]
// CHECK:STDOUT:     %Create.ref: @F.%Create.type (%Create.type.ff1) = name_ref Create, %.loc21_7 [symbolic = %Create (constants.%Create.96b)]
// CHECK:STDOUT:     %.loc21_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %Create.specific_fn.loc21_7.1: <specific function> = specific_function %Create.ref, @Create(constants.%V.as_type) [symbolic = %Create.specific_fn.loc21_7.2 (constants.%Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc21_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V.as_type)> [symbolic = %ImplicitAs.type.loc21_16.2 (constants.%ImplicitAs.type.113)]
// CHECK:STDOUT:     %.loc21_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%V.as_type) [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = name_ref Convert, %.loc21_16.2 [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:     %.loc21_16.3: @F.%V.as_type.loc21_6.2 (%V.as_type) = converted %.loc21_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %Create.call: init %empty_tuple.type = call %Create.specific_fn.loc21_7.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Create.type => constants.%Create.type.f31
// CHECK:STDOUT:   %Create => constants.%Create.cc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc10_6.2 => constants.%U
// CHECK:STDOUT:   %V.loc10_16.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V.as_type) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%V.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Create.type => constants.%Create.type.ff1
// CHECK:STDOUT:   %Create => constants.%Create.96b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Create(constants.%V.as_type) {
// CHECK:STDOUT:   %T => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6e0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
