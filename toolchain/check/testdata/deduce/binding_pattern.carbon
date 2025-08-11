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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %C.Create.type.f31: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.cc8: %C.Create.type.f31 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [symbolic]
// CHECK:STDOUT:   %pattern_type.1d2: type = pattern_type %ptr.7d2 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5e5: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.type.beb: type = fn_type @C.Create, @C(%V) [symbolic]
// CHECK:STDOUT:   %C.Create.dca: %C.Create.type.beb = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.d93: <witness> = require_complete_type %C.5e5 [symbolic]
// CHECK:STDOUT:   %pattern_type.a32: type = pattern_type %V [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.dca, @C.Create(%V) [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d88: type = facet_type <@ImplicitAs, @ImplicitAs(%V)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.5fe: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V) [symbolic]
// CHECK:STDOUT:   %assoc0.a0b: %ImplicitAs.assoc_type.5fe = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.6c8: <witness> = require_complete_type %ImplicitAs.type.d88 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude/parts/as, loc12_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//prelude/parts/as, loc12_35, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude/parts/as, loc12_35, unloaded
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
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc8_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_6.1 (constants.%U)]
// CHECK:STDOUT:     %V.loc8_16.2: type = bind_symbolic_name V, 1 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%C.f2e as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = value_param call_param0
// CHECK:STDOUT:       %.loc4_19.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.f2e)] {
// CHECK:STDOUT:         %.loc4_19.3: type = specific_constant constants.%C.f2e, @C(constants.%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_19.3 [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.f31)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.f31) = struct_value () [symbolic = %C.Create (constants.%C.Create.cc8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.f31) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.cc8)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = bind_name value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%value.param: @C.Create.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1d2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%U.loc8_6.2: type, %V.loc8_16.2: type) {
// CHECK:STDOUT:   %U.loc8_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_6.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc8_16.1: type = bind_symbolic_name V, 1 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc19_6.2: type = class_type @C, @C(%V.loc8_16.1) [symbolic = %C.loc19_6.2 (constants.%C.5e5)]
// CHECK:STDOUT:   %require_complete.loc19_7: <witness> = require_complete_type %C.loc19_6.2 [symbolic = %require_complete.loc19_7 (constants.%require_complete.d93)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.loc8_16.1) [symbolic = %C.Create.type (constants.%C.Create.type.beb)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.beb) = struct_value () [symbolic = %C.Create (constants.%C.Create.dca)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc19_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.loc8_16.1) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc19_16.1: <witness> = require_complete_type %V.loc8_16.1 [symbolic = %require_complete.loc19_16.1 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %ImplicitAs.type.loc19_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.loc8_16.1)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.d88)]
// CHECK:STDOUT:   %require_complete.loc19_16.2: <witness> = require_complete_type %ImplicitAs.type.loc19_16.2 [symbolic = %require_complete.loc19_16.2 (constants.%require_complete.6c8)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.loc8_16.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.5fe)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.a0b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc8_16.2 [symbolic = %V.loc8_16.1 (constants.%V)]
// CHECK:STDOUT:     %C.loc19_6.1: type = class_type @C, @C(constants.%V) [symbolic = %C.loc19_6.2 (constants.%C.5e5)]
// CHECK:STDOUT:     %.loc19_7: @F.%C.Create.type (%C.Create.type.beb) = specific_constant @C.%C.Create.decl, @C(constants.%V) [symbolic = %C.Create (constants.%C.Create.dca)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.beb) = name_ref Create, %.loc19_7 [symbolic = %C.Create (constants.%C.Create.dca)]
// CHECK:STDOUT:     %.loc19_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %C.Create.specific_fn.loc19_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V) [symbolic = %C.Create.specific_fn.loc19_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc19_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V)> [symbolic = %ImplicitAs.type.loc19_16.2 (constants.%ImplicitAs.type.d88)]
// CHECK:STDOUT:     %.loc19_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%V) [symbolic = %assoc0 (constants.%assoc0.a0b)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = name_ref Convert, %.loc19_16.2 [symbolic = %assoc0 (constants.%assoc0.a0b)]
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.f31
// CHECK:STDOUT:   %C.Create => constants.%C.Create.cc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.f2e
// CHECK:STDOUT:   %ptr => constants.%ptr.7d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1d2
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
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.beb
// CHECK:STDOUT:   %C.Create => constants.%C.Create.dca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
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
// CHECK:STDOUT:   %C.Create.type.f31: type = fn_type @C.Create, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.Create.cc8: %C.Create.type.f31 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [symbolic]
// CHECK:STDOUT:   %pattern_type.1d2: type = pattern_type %ptr.7d2 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.aba: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self)> [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %V: %type_where = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.344: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %C.cd6: type = class_type @C, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.type.ff1: type = fn_type @C.Create, @C(%V.as_type) [symbolic]
// CHECK:STDOUT:   %C.Create.96b: %C.Create.type.ff1 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.558: <witness> = require_complete_type %C.cd6 [symbolic]
// CHECK:STDOUT:   %pattern_type.6e0: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %C.Create.specific_fn: <specific function> = specific_function %C.Create.96b, @C.Create(%V.as_type) [symbolic]
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
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude/parts/as, loc12_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//prelude/parts/as, loc12_35, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude/parts/as, loc12_35, unloaded
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
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.344 = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc9_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc9_6.1 (constants.%U)]
// CHECK:STDOUT:     %.loc9_25.1: type = splice_block %.loc9_25.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %ImplicitAs.type.loc9: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self)> [symbolic_self = constants.%ImplicitAs.type.aba]
// CHECK:STDOUT:       %.loc9_32.2: type = converted %.loc9_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_25.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_base_facet_type type
// CHECK:STDOUT:         requirement_impls %.loc9_32.2, %ImplicitAs.type.loc9
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_16.2: %type_where = bind_symbolic_name V, 1 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%C.f2e as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = value_param call_param0
// CHECK:STDOUT:       %.loc4_19.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.f2e)] {
// CHECK:STDOUT:         %.loc4_19.3: type = specific_constant constants.%C.f2e, @C(constants.%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_19.3 [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%T.loc4_9.1) [symbolic = %C.Create.type (constants.%C.Create.type.f31)]
// CHECK:STDOUT:   %C.Create: @C.%C.Create.type (%C.Create.type.f31) = struct_value () [symbolic = %C.Create (constants.%C.Create.cc8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.Create.decl: @C.%C.Create.type (%C.Create.type.f31) = fn_decl @C.Create [symbolic = @C.%C.Create (constants.%C.Create.cc8)] {
// CHECK:STDOUT:       %value.patt: @C.Create.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern value [concrete]
// CHECK:STDOUT:       %value.param_patt: @C.Create.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %value.param: @C.Create.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %value: @C.Create.%T (%T) = bind_name value, %value.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Create = %C.Create.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.Create(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%value.param: @C.Create.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1d2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%U.loc9_6.2: type, %V.loc9_16.2: %type_where) {
// CHECK:STDOUT:   %U.loc9_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc9_6.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc9_16.1: %type_where = bind_symbolic_name V, 1 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %V.as_type.loc20_6.2: type = facet_access_type %V.loc9_16.1 [symbolic = %V.as_type.loc20_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %C.loc20_6.2: type = class_type @C, @C(%V.as_type.loc20_6.2) [symbolic = %C.loc20_6.2 (constants.%C.cd6)]
// CHECK:STDOUT:   %require_complete.loc20_7: <witness> = require_complete_type %C.loc20_6.2 [symbolic = %require_complete.loc20_7 (constants.%require_complete.558)]
// CHECK:STDOUT:   %C.Create.type: type = fn_type @C.Create, @C(%V.as_type.loc20_6.2) [symbolic = %C.Create.type (constants.%C.Create.type.ff1)]
// CHECK:STDOUT:   %C.Create: @F.%C.Create.type (%C.Create.type.ff1) = struct_value () [symbolic = %C.Create (constants.%C.Create.96b)]
// CHECK:STDOUT:   %C.Create.specific_fn.loc20_7.2: <specific function> = specific_function %C.Create, @C.Create(%V.as_type.loc20_6.2) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc20_16.1: <witness> = require_complete_type %V.as_type.loc20_6.2 [symbolic = %require_complete.loc20_16.1 (constants.%require_complete.0b5)]
// CHECK:STDOUT:   %ImplicitAs.type.loc20_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%V.as_type.loc20_6.2)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.113)]
// CHECK:STDOUT:   %require_complete.loc20_16.2: <witness> = require_complete_type %ImplicitAs.type.loc20_16.2 [symbolic = %require_complete.loc20_16.2 (constants.%require_complete.9fe)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%V.as_type.loc20_6.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.83a)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %V.ref: %type_where = name_ref V, %V.loc9_16.2 [symbolic = %V.loc9_16.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc20_6.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc20_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc20_6: type = converted %V.ref, %V.as_type.loc20_6.1 [symbolic = %V.as_type.loc20_6.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %C.loc20_6.1: type = class_type @C, @C(constants.%V.as_type) [symbolic = %C.loc20_6.2 (constants.%C.cd6)]
// CHECK:STDOUT:     %.loc20_7: @F.%C.Create.type (%C.Create.type.ff1) = specific_constant @C.%C.Create.decl, @C(constants.%V.as_type) [symbolic = %C.Create (constants.%C.Create.96b)]
// CHECK:STDOUT:     %Create.ref: @F.%C.Create.type (%C.Create.type.ff1) = name_ref Create, %.loc20_7 [symbolic = %C.Create (constants.%C.Create.96b)]
// CHECK:STDOUT:     %.loc20_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %C.Create.specific_fn.loc20_7.1: <specific function> = specific_function %Create.ref, @C.Create(constants.%V.as_type) [symbolic = %C.Create.specific_fn.loc20_7.2 (constants.%C.Create.specific_fn)]
// CHECK:STDOUT:     %ImplicitAs.type.loc20_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%V.as_type)> [symbolic = %ImplicitAs.type.loc20_16.2 (constants.%ImplicitAs.type.113)]
// CHECK:STDOUT:     %.loc20_16.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%V.as_type) [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.83a) = name_ref Convert, %.loc20_16.2 [symbolic = %assoc0 (constants.%assoc0.5dc)]
// CHECK:STDOUT:     %.loc20_16.3: @F.%V.as_type.loc20_6.2 (%V.as_type) = converted %.loc20_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.Create.call: init %empty_tuple.type = call %C.Create.specific_fn.loc20_7.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.f31
// CHECK:STDOUT:   %C.Create => constants.%C.Create.cc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.f2e
// CHECK:STDOUT:   %ptr => constants.%ptr.7d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U, constants.%V) {
// CHECK:STDOUT:   %U.loc9_6.1 => constants.%U
// CHECK:STDOUT:   %V.loc9_16.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V.as_type) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%V.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.Create.type => constants.%C.Create.type.ff1
// CHECK:STDOUT:   %C.Create => constants.%C.Create.96b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Create(constants.%V.as_type) {
// CHECK:STDOUT:   %T => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6e0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
