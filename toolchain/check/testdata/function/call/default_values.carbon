// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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
// CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE+4]]:63: error: cannot initialize tuple of 2 elements from tuple with 0 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ()) {}
// CHECK:STDERR:                                                               ^~
// CHECK:STDERR:
fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ()) {}

fn G() {
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE+14]]:3: error: 4 arguments passed to function expecting 3 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   F({}, {}, {}, {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-11]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(unused x: C = {}, unused y: C = {}, unused z: C = {}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-10]]:24: error: tuple pattern expects 2 elements, but tuple literal has 3 [TuplePatternSizeDoesntMatchLiteral]
  // CHECK:STDERR: fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ()) {}
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_caller_too_many_arguments.carbon:[[@LINE-13]]:24: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn H(unused x: C = {}, (unused y: C = {}, unused z: C = {}) = ()) {}
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

fn G() {
  //@dump-sem-ir-begin
  F({});
  //@dump-sem-ir-end
}

// --- tuple_defaults_fully_specified.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C, (unused y: C, unused z: C) = ({}, {}), unused w: C = {}) {}

fn G() {
  //@dump-sem-ir-begin
  F({});
  //@dump-sem-ir-end
}

// --- single_argument.carbon
library "[[@TEST_NAME]]";
class C {}

fn F(unused x: C = {}) {}

fn G() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- imported_callee.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_6.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound: <bound method> = bound_method %.115, %Destroy.WithSelf.Op.403171.2 [concrete]
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc8_6.6, @F.%.loc4_55.8, %tuple.elem0, %tuple.elem1)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound(constants.%.115)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(@F.%.loc4_21.6)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
