// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/default_values.carbon

// --- fail_caller_too_many_arguments.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C = {}, unused y: C = {}, unused z: C = {}) {}
fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ({}, {})) {}

fn G() {
  // CHECK:STDERR:        ╭─┤ fail_caller_too_many_arguments.carbon:[[@LINE+22]]:3
  // CHECK:STDERR:        │ fn F(unused x: C = {}, unused y: C = {}, unused z: C = {}) {}
  // CHECK:STDERR:        · ──────────────────────────────┬─────────────────────────────
  // CHECK:STDERR:        ·                               ╰──┤ calling function declared here
  // CHECK:STDERR:        ·                                  │ [InCallToEntity]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F({}, {}, {}, {});
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: 4 arguments passed to function expecting 3 arguments
  // CHECK:STDERR:        ·              ┃        [CallArgCountMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_caller_too_many_arguments.carbon:[[@LINE-15]]:24
  // CHECK:STDERR: ->     │ fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ({}, {})) {}
  // CHECK:STDERR:        ·                        ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━───────────
  // CHECK:STDERR:        ·                        ┗━━┫ error: tuple pattern expects 2 elements, but tuple
  // CHECK:STDERR:        ·                        │  ┃        literal has 3
  // CHECK:STDERR:        ·                        │  ┃        [TuplePatternSizeDoesntMatchLiteral]
  // CHECK:STDERR:        ·                        ╰────┤ initializing function parameter
  // CHECK:STDERR:        ·                             │ [InCallToFunctionParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F({}, {}, {}, {});
  H({}, ({}, {}, {}));
}

// --- fail_caller_too_few_arguments.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C, unused y: C = {}) {}

fn G() {
  // CHECK:STDERR:        ╭─┤ fail_caller_too_few_arguments.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │ fn F(unused x: C, unused y: C = {}) {}
  // CHECK:STDERR:        · ──────────────────┬──────────────────
  // CHECK:STDERR:        ·                   ╰──┤ calling function declared here [InCallToEntity]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: 0 arguments passed to function expecting 2 arguments
  // CHECK:STDERR:        ·       ┃        [CallArgCountMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F();
}

// --- imported_callee.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(x: C, y: C = {});

// --- imported_callee.impl.carbon
impl library "[[@TEST_NAME]]";

fn F(unused x: C, unused y: C = {}) {}

//@dump-sem-ir-begin
fn G() {
  F({});
}
//@dump-sem-ir-end


// --- tuple_defaults_fully_specified.carbon
library "[[@TEST_NAME]]";
class C {}

//@dump-sem-ir-begin
fn F(unused x: C, (unused y: C, unused z: C) = ({}, {}), unused w: C = {}) {}

fn G() {
  F({});
}
//@dump-sem-ir-end

// --- single_argument.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C = {}) {}

//@dump-sem-ir-begin
fn G() {
  F();
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- imported_callee.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_6.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc7_6.3: init %C to %.loc7_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_6.4: init %C = converted %.loc7_6.1, %.loc7_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_6.5: ref %C = temporary %.loc7_6.2, %.loc7_6.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc7_6.6: %C = acquire_value %.loc7_6.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc7_6.6, @F.%.loc3_34.6)
// CHECK:STDOUT:   %Destroy.facet.loc7_6.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_6.7: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_6.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_6.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_6.8: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_6.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc7_6.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_6.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_6.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_defaults_fully_specified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.98b = wrapper_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %z.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.98b = wrapper_binding_pattern z, %z.param_patt [concrete]
// CHECK:STDOUT:   %tuple.type.748: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %tuple.type.748 [concrete]
// CHECK:STDOUT:   %.2bb: %pattern_type.8b4 = tuple_pattern (%y.patt, %z.patt) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.9a3: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %tuple.d3e: %tuple.type.748 = tuple_value (%C.val, %C.val) [concrete]
// CHECK:STDOUT:   %.401: %pattern_type.8b4 = default_value_pattern %.2bb, %tuple.d3e [concrete]
// CHECK:STDOUT:   %w.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.98b = wrapper_binding_pattern w, %w.param_patt [concrete]
// CHECK:STDOUT:   %.f96: %pattern_type.98b = default_value_pattern %w.patt, %C.val [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_6.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.98b = wrapper_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:     %z.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%z.param_patt]
// CHECK:STDOUT:     %z.patt: %pattern_type.98b = wrapper_binding_pattern z, %z.param_patt [concrete = constants.%z.patt]
// CHECK:STDOUT:     %.loc5_44: %pattern_type.8b4 = tuple_pattern (%y.patt, %z.patt) [concrete = constants.%.2bb]
// CHECK:STDOUT:     %.loc5_46: %pattern_type.8b4 = default_value_pattern %.loc5_44, %.loc5_55.8 [concrete = constants.%.401]
// CHECK:STDOUT:     %w.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%w.param_patt]
// CHECK:STDOUT:     %w.patt: %pattern_type.98b = wrapper_binding_pattern w, %w.param_patt [concrete = constants.%w.patt]
// CHECK:STDOUT:     %.loc5_70: %pattern_type.98b = default_value_pattern %w.patt, %.loc5_73.6 [concrete = constants.%.f96]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_54.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_55.1: %tuple.type.b6b = tuple_literal (%.loc5_50.1, %.loc5_54.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:     %.loc5_50.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc5_50.3: init %C to %.loc5_50.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_55.2: init %C = converted %.loc5_50.1, %.loc5_50.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_55.3: ref %C = temporary %.loc5_50.2, %.loc5_55.2 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc5_55.4: %C = acquire_value %.loc5_55.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_54.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc5_54.3: init %C to %.loc5_54.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_55.5: init %C = converted %.loc5_54.1, %.loc5_54.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_55.6: ref %C = temporary %.loc5_54.2, %.loc5_55.5 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc5_55.7: %C = acquire_value %.loc5_55.6 [concrete = constants.%C.val]
// CHECK:STDOUT:     %tuple: %tuple.type.748 = tuple_value (%.loc5_55.4, %.loc5_55.7) [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %.loc5_55.8: %tuple.type.748 = converted %.loc5_55.1, %tuple [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %.loc5_73.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_73.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc5_73.3: init %C to %.loc5_73.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_73.4: init %C = converted %.loc5_73.1, %.loc5_73.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_73.5: ref %C = temporary %.loc5_73.2, %.loc5_73.4 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc5_73.6: %C = acquire_value %.loc5_73.5 [concrete = constants.%C.val]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc5_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc5_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %y: %C = wrapper_binding y, %y.param
// CHECK:STDOUT:     %z.param: %C = value_param call_param2
// CHECK:STDOUT:     %C.ref.loc5_43: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %z: %C = wrapper_binding z, %z.param
// CHECK:STDOUT:     %w.param: %C = value_param call_param3
// CHECK:STDOUT:     %C.ref.loc5_68: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %w: %C = wrapper_binding w, %w.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %C, %y.param: %C, %z.param: %C, %w.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc8_6.3: init %C to %.loc8_6.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_6.4: init %C = converted %.loc8_6.1, %.loc8_6.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_6.5: ref %C = temporary %.loc8_6.2, %.loc8_6.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc8_6.6: %C = acquire_value %.loc8_6.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %tuple.elem0: %C = tuple_access @F.%.loc5_55.8, element0 [concrete = constants.%C.val]
// CHECK:STDOUT:   %tuple.elem1: %C = tuple_access @F.%.loc5_55.8, element1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc8_6.6, %tuple.elem0, %tuple.elem1, @F.%.loc5_73.6)
// CHECK:STDOUT:   %Destroy.facet.loc8_6.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_6.7: %Destroy.type = converted constants.%C, %Destroy.facet.loc8_6.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_6.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_6.8: %Destroy.type = converted constants.%C, %Destroy.facet.loc8_6.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_6.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_6.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_6.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- single_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(@F.%.loc4_21.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
