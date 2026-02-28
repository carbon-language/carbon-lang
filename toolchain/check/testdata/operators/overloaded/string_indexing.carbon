// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon

// --- fail_literal_negative_index.carbon
library "[[@TEST_NAME]]";

fn TestLiteralNegativeIndex() {
  // CHECK:STDERR: fail_literal_negative_index.carbon:[[@LINE+4]]:24: error: index `-1` is negative. [StringAtIndexNegative]
  // CHECK:STDERR:   let unused c: char = "Test"[-1];
  // CHECK:STDERR:                        ^~~~~~~~~~
  // CHECK:STDERR:
  let unused c: char = "Test"[-1];
}

// --- fail_literal_out_of_bounds.carbon
library "[[@TEST_NAME]]";

fn TestLiteralOutOfBounds() {
  // CHECK:STDERR: fail_literal_out_of_bounds.carbon:[[@LINE+4]]:24: error: string index `4` is out of bounds; string has length 4. [StringAtIndexOutOfBounds]
  // CHECK:STDERR:   let unused c: char = "Test"[4];
  // CHECK:STDERR:                        ^~~~~~~~~
  // CHECK:STDERR:
  let unused c: char = "Test"[4];
}

// --- fail_wrong_type.carbon
library "[[@TEST_NAME]]";

fn TestWrongType() {
  class C{}
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.IndexWith(type)` in type `str` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   "Test"[C];
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  "Test"[C];
  //@dump-sem-ir-end
}

// --- fail_bad_decl.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "string.at" [InvalidBuiltinSignature]
// CHECK:STDERR: fn At(s: str, index: i64) -> i32 = "string.at";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn At(s: str, index: i64) -> i32 = "string.at";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "string.at" [InvalidBuiltinSignature]
// CHECK:STDERR: fn At2(s: str) -> char = "string.at";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn At2(s: str) -> char = "string.at";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "string.at" [InvalidBuiltinSignature]
// CHECK:STDERR: fn At3(s: i32, index: i64) -> char = "string.at";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn At3(s: i32, index: i64) -> char = "string.at";

// --- test_string_indexing.carbon
library "[[@TEST_NAME]]";

fn TestStringIndexing() {
  //@dump-sem-ir-begin
  let unused c: char = "Test"[0];
  let unused d: char = "Test"[3];
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_wrong_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %str.0a6: %ptr.fb0 = string_literal "Test" [concrete]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.0a6, %int_4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestWrongType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %str: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_4) [concrete = constants.%String.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- test_string_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %str.0a6: %ptr.fb0 = string_literal "Test" [concrete]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.0a6, %int_4) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IndexWith.type.d54: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %T.1a9: %ImplicitAs.type.2ad = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.85b: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%T.1a9) [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.4fc: %str.as.IndexWith.impl.At.type.85b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4e6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.373, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.4e6) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.8d4: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.595: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.a70: %str.as.IndexWith.impl.At.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.d54 = facet_value %str.ee0, (%IndexWith.impl_witness.8d4) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.0ac: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(Core.IntLiteral, %IndexWith.facet) [concrete]
// CHECK:STDOUT:   %.b7b: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.0ac, %IndexWith.facet [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.bound: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.a70 [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %str.as.IndexWith.impl.At.a70, @str.as.IndexWith.impl.At(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.specific_fn [concrete]
// CHECK:STDOUT:   %int_84: %char = int_value 84 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_116: %char = int_value 116 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.aa4 = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0ce8: @str.as.IndexWith.impl.%str.as.IndexWith.impl.At.type (%str.as.IndexWith.impl.At.type.85b) = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, loaded [symbolic = @str.as.IndexWith.impl.%str.as.IndexWith.impl.At (constants.%str.as.IndexWith.impl.At.4fc)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.aa4, %Core.import_ref.0ce8), @str.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.8ed = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.373 = impl_witness_table (%Core.import_ref.8ed), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStringIndexing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str.loc5: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4.loc5: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val.loc5: %str.ee0 = struct_value (%str.loc5, %int_4.loc5) [concrete = constants.%String.val]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem1.loc5: %.b7b = impl_witness_access constants.%IndexWith.impl_witness.8d4, element1 [concrete = constants.%str.as.IndexWith.impl.At.a70]
// CHECK:STDOUT:   %bound_method.loc5_32.1: <bound method> = bound_method %String.val.loc5, %impl.elem1.loc5 [concrete = constants.%str.as.IndexWith.impl.At.bound]
// CHECK:STDOUT:   %ImplicitAs.facet.loc5_32.1: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.4e6) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc5_32.1: %ImplicitAs.type.2ad = converted Core.IntLiteral, %ImplicitAs.facet.loc5_32.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc5_32.2: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.4e6) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc5_32.2: %ImplicitAs.type.2ad = converted Core.IntLiteral, %ImplicitAs.facet.loc5_32.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem1.loc5, @str.as.IndexWith.impl.At(constants.%ImplicitAs.facet) [concrete = constants.%str.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_32.2: <bound method> = bound_method %String.val.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.loc5: init %char = call %bound_method.loc5_32.2(%String.val.loc5, %int_0) [concrete = constants.%int_84]
// CHECK:STDOUT:   %char.loc5: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %.loc5_32.3: %char = value_of_initializer %str.as.IndexWith.impl.At.call.loc5 [concrete = constants.%int_84]
// CHECK:STDOUT:   %.loc5_32.4: %char = converted %str.as.IndexWith.impl.At.call.loc5, %.loc5_32.3 [concrete = constants.%int_84]
// CHECK:STDOUT:   %c: %char = value_binding c, %.loc5_32.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b09 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str.loc6: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4.loc6: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val.loc6: %str.ee0 = struct_value (%str.loc6, %int_4.loc6) [concrete = constants.%String.val]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %impl.elem1.loc6: %.b7b = impl_witness_access constants.%IndexWith.impl_witness.8d4, element1 [concrete = constants.%str.as.IndexWith.impl.At.a70]
// CHECK:STDOUT:   %bound_method.loc6_32.1: <bound method> = bound_method %String.val.loc6, %impl.elem1.loc6 [concrete = constants.%str.as.IndexWith.impl.At.bound]
// CHECK:STDOUT:   %ImplicitAs.facet.loc6_32.1: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.4e6) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc6_32.1: %ImplicitAs.type.2ad = converted Core.IntLiteral, %ImplicitAs.facet.loc6_32.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc6_32.2: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.4e6) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc6_32.2: %ImplicitAs.type.2ad = converted Core.IntLiteral, %ImplicitAs.facet.loc6_32.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem1.loc6, @str.as.IndexWith.impl.At(constants.%ImplicitAs.facet) [concrete = constants.%str.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_32.2: <bound method> = bound_method %String.val.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.loc6: init %char = call %bound_method.loc6_32.2(%String.val.loc6, %int_3) [concrete = constants.%int_116]
// CHECK:STDOUT:   %char.loc6: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %.loc6_32.3: %char = value_of_initializer %str.as.IndexWith.impl.At.call.loc6 [concrete = constants.%int_116]
// CHECK:STDOUT:   %.loc6_32.4: %char = converted %str.as.IndexWith.impl.At.call.loc6, %.loc6_32.3 [concrete = constants.%int_116]
// CHECK:STDOUT:   %d: %char = value_binding d, %.loc6_32.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
