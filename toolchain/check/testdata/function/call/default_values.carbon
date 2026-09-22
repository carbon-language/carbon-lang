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
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE+14]]:3: error: 4 arguments passed to function expecting 3 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   F({}, {}, {}, {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-7]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(unused x: C = {}, unused y: C = {}, unused z: C = {}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-10]]:24: error: tuple pattern expects 2 elements, but tuple literal has 3 [TuplePatternSizeDoesntMatchLiteral]
  // CHECK:STDERR: fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ({}, {})) {}
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-13]]:24: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ({}, {})) {}
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F({}, {}, {}, {});
  H({}, ({}, {}, {}));
}

// --- fail_caller_too_few_arguments.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C, unused y: C = {}) {}

fn G() {
  // CHECK:STDERR: fail_caller_too_few_arguments.carbon:[[@LINE+7]]:3: error: 0 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_caller_too_few_arguments.carbon:[[@LINE-6]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(unused x: C, unused y: C = {}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F();
}

// --- imported_callee.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(x: C, y: C = {});

// --- imported_callee.impl.carbon
impl library "[[@TEST_NAME]]";

fn F(unused x: C, unused y: C = _) {}

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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc7_6.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound: <bound method> = bound_method %.115, %Destroy.WithSelf.Op.403171.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.d96: %C = import_ref Main//imported_callee, loc4_20, loaded [concrete = constants.%C.val]
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
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc7_6.6, imports.%Main.import_ref.d96)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_6.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_6.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %.e69: %pattern_type.8b4 = default_value_pattern %.2bb, @F.%.loc5_55.8 [concrete]
// CHECK:STDOUT:   %w.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.98b = wrapper_binding_pattern w, %w.param_patt [concrete]
// CHECK:STDOUT:   %.692: %pattern_type.98b = default_value_pattern %w.patt, @F.%.loc5_73.6 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %tuple.d3e: %tuple.type.748 = tuple_value (%C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_6.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound: <bound method> = bound_method %.115, %Destroy.WithSelf.Op.403171.2 [concrete]
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
// CHECK:STDOUT:     %.loc5_46: %pattern_type.8b4 = default_value_pattern %.loc5_44, %.loc5_55.8 [concrete = constants.%.e69]
// CHECK:STDOUT:     %w.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%w.param_patt]
// CHECK:STDOUT:     %w.patt: %pattern_type.98b = wrapper_binding_pattern w, %w.param_patt [concrete = constants.%w.patt]
// CHECK:STDOUT:     %.loc5_70: %pattern_type.98b = default_value_pattern %w.patt, %.loc5_73.6 [concrete = constants.%.692]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_54.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_55.1: %tuple.type.b6b = tuple_literal (%.loc5_50.1, %.loc5_54.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:     %.loc5_73.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc5_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc5_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %y: %C = wrapper_binding y, %y.param
// CHECK:STDOUT:     %z.param: %C = value_param call_param2
// CHECK:STDOUT:     %C.ref.loc5_43: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %z: %C = wrapper_binding z, %z.param
// CHECK:STDOUT:     %tuple.loc5_44: %tuple.type.748 = tuple_value (%y.param, %z.param)
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
// CHECK:STDOUT:     %tuple.loc5_55: %tuple.type.748 = tuple_value (%.loc5_55.4, %.loc5_55.7) [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %.loc5_55.8: %tuple.type.748 = converted %.loc5_55.1, %tuple.loc5_55 [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %w.param: %C = value_param call_param3
// CHECK:STDOUT:     %C.ref.loc5_68: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %w: %C = wrapper_binding w, %w.param
// CHECK:STDOUT:     %.loc5_73.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc5_73.3: init %C to %.loc5_73.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_73.4: init %C = converted %.loc5_73.1, %.loc5_73.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc5_73.5: ref %C = temporary %.loc5_73.2, %.loc5_73.4 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc5_73.6: %C = acquire_value %.loc5_73.5 [concrete = constants.%C.val]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_6.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_6.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
