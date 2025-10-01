// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/type/can_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/type/can_destroy.carbon

// --- param.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";

fn F(T:! CanDestroy()) {}

fn G() {
  //@dump-sem-ir-begin
  F(());
  F({});
  //@dump-sem-ir-end
}

// --- impls.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";

fn F(T:! type where .Self impls CanDestroy()) {}

fn G() {
  //@dump-sem-ir-begin
  F(());
  F({});
  //@dump-sem-ir-end
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";

fn F(T:! CanDestroy()) {}

class Incomplete;

fn G() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:3: error: cannot convert type `Incomplete` into type implementing `type where .Self impls Core.CanDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(Incomplete);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! CanDestroy()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(Incomplete);
}

// --- fail_abstract.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";

fn F(T:! CanDestroy()) {}

abstract class Abstract {}

fn G() {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:3: error: cannot convert type `Abstract` into type implementing `type where .Self impls Core.CanDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F(Abstract);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! CanDestroy()) {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(Abstract);
}

// --- fail_impl.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";
fn TypeAnd(a: type, b: type) -> type = "type.and";

class C {}

interface I {}

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl C as CanDestroy() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as CanDestroy() {}

// --- impl_with_interface.carbon
library "[[@TEST_NAME]]";

fn CanDestroy() -> type = "type.can_destroy";
fn TypeAnd(a: type, b: type) -> type = "type.and";

class C {}

interface I {}

impl C as TypeAnd(I, CanDestroy()) {}

// CHECK:STDOUT: --- param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.ff9: %type_where = facet_value %empty_tuple.type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.1af: <specific function> = specific_function %F, @F(%facet_value.ff9) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.da1: <specific function> = specific_function %F, @F(%facet_value.7c2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %facet_value.loc9: %type_where = facet_value constants.%empty_tuple.type, () [concrete = constants.%facet_value.ff9]
// CHECK:STDOUT:   %.loc9_7: %type_where = converted %.loc9_6, %facet_value.loc9 [concrete = constants.%facet_value.ff9]
// CHECK:STDOUT:   %F.specific_fn.loc9: <specific function> = specific_function %F.ref.loc9, @F(constants.%facet_value.ff9) [concrete = constants.%F.specific_fn.1af]
// CHECK:STDOUT:   %F.call.loc9: init %empty_tuple.type = call %F.specific_fn.loc9()
// CHECK:STDOUT:   %F.ref.loc10: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %facet_value.loc10: %type_where = facet_value constants.%empty_struct_type, () [concrete = constants.%facet_value.7c2]
// CHECK:STDOUT:   %.loc10_7: %type_where = converted %.loc10_6, %facet_value.loc10 [concrete = constants.%facet_value.7c2]
// CHECK:STDOUT:   %F.specific_fn.loc10: <specific function> = specific_function %F.ref.loc10, @F(constants.%facet_value.7c2) [concrete = constants.%F.specific_fn.da1]
// CHECK:STDOUT:   %F.call.loc10: init %empty_tuple.type = call %F.specific_fn.loc10()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.ff9: %type_where = facet_value %empty_tuple.type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.1af: <specific function> = specific_function %F, @F(%facet_value.ff9) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.da1: <specific function> = specific_function %F, @F(%facet_value.7c2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %facet_value.loc9: %type_where = facet_value constants.%empty_tuple.type, () [concrete = constants.%facet_value.ff9]
// CHECK:STDOUT:   %.loc9_7: %type_where = converted %.loc9_6, %facet_value.loc9 [concrete = constants.%facet_value.ff9]
// CHECK:STDOUT:   %F.specific_fn.loc9: <specific function> = specific_function %F.ref.loc9, @F(constants.%facet_value.ff9) [concrete = constants.%F.specific_fn.1af]
// CHECK:STDOUT:   %F.call.loc9: init %empty_tuple.type = call %F.specific_fn.loc9()
// CHECK:STDOUT:   %F.ref.loc10: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %facet_value.loc10: %type_where = facet_value constants.%empty_struct_type, () [concrete = constants.%facet_value.7c2]
// CHECK:STDOUT:   %.loc10_7: %type_where = converted %.loc10_6, %facet_value.loc10 [concrete = constants.%facet_value.7c2]
// CHECK:STDOUT:   %F.specific_fn.loc10: <specific function> = specific_function %F.ref.loc10, @F(constants.%facet_value.7c2) [concrete = constants.%F.specific_fn.da1]
// CHECK:STDOUT:   %F.call.loc10: init %empty_tuple.type = call %F.specific_fn.loc10()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
