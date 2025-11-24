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
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_literal_negative_index.carbon:[[@LINE+4]]:17: error: index `-1` is negative. [StringIndexNegative]
  // CHECK:STDERR:   let c: char = "Test"[-1];
  // CHECK:STDERR:                 ^~~~~~~~~~
  // CHECK:STDERR:
  let c: char = "Test"[-1];
  //@dump-sem-ir-end
}

// --- fail_literal_out_of_bounds.carbon
library "[[@TEST_NAME]]";

fn TestLiteralOutOfBounds() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_literal_out_of_bounds.carbon:[[@LINE+4]]:17: error: string index `4` is out of bounds; string has length 4. [StringAtIndexOutOfBounds]
  // CHECK:STDERR:   let c: char = "Test"[4];
  // CHECK:STDERR:                 ^~~~~~~~~
  // CHECK:STDERR:
  let c: char = "Test"[4];
  //@dump-sem-ir-end
}

// --- fail_wrong_type.carbon
library "[[@TEST_NAME]]";

fn TestWrongType() {
  let x: i32 = 42;
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:17: error: cannot access member of interface `Core.IndexWith(Core.IntLiteral)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c: char = x[0];
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  let c: char = x[0];
  //@dump-sem-ir-end
}

// --- test_string_indexing.carbon
library "[[@TEST_NAME]]";

fn TestStringIndexing() {
  //@dump-sem-ir-begin
  let c: char = "Test"[0];
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_literal_negative_index.carbon
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.6a1: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %IndexWith.type.8ab: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %IndexWith.At.type.1ab: type = fn_type @IndexWith.At, @IndexWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %T.6be: %ImplicitAs.type.d14 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.4f6: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%T.6be) [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.362: %str.as.IndexWith.impl.At.type.4f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3ced: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.462, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3ced) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.4ec: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.1d2: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.17e: %str.as.IndexWith.impl.At.type.1d2 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.8ab = facet_value %str.ee0, (%IndexWith.impl_witness.4ec) [concrete]
// CHECK:STDOUT:   %.d99: type = fn_type_with_self_type %IndexWith.At.type.1ab, %IndexWith.facet [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.bound: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.17e [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %str.as.IndexWith.impl.At.17e, @str.as.IndexWith.impl.At(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b9e: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.b9e), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.484 = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.14a: @str.as.IndexWith.impl.%str.as.IndexWith.impl.At.type (%str.as.IndexWith.impl.At.type.4f6) = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, loaded [symbolic = @str.as.IndexWith.impl.%str.as.IndexWith.impl.At (constants.%str.as.IndexWith.impl.At.362)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.484, %Core.import_ref.14a), @str.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.8f7 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.462 = impl_witness_table (%Core.import_ref.8f7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLiteralNegativeIndex() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_4) [concrete = constants.%String.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1.loc9_24: %.6a1 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_24: <bound method> = bound_method %int_1, %impl.elem1.loc9_24 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc9_24(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem1.loc9_26: %.d99 = impl_witness_access constants.%IndexWith.impl_witness.4ec, element1 [concrete = constants.%str.as.IndexWith.impl.At.17e]
// CHECK:STDOUT:   %bound_method.loc9_26.1: <bound method> = bound_method %String.val, %impl.elem1.loc9_26 [concrete = constants.%str.as.IndexWith.impl.At.bound]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_26.1: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_26.1: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_26.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_26.2: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_26.2: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_26.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1.loc9_26, @str.as.IndexWith.impl.At(constants.%ImplicitAs.facet) [concrete = constants.%str.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_26.2: <bound method> = bound_method %String.val, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %.loc9_24.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc9_24.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc9_24.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call: init %char = call %bound_method.loc9_26.2(%String.val, %.loc9_24.2)
// CHECK:STDOUT:   %.loc9_26.3: %char = value_of_initializer %str.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc9_26.4: %char = converted %str.as.IndexWith.impl.At.call, %.loc9_26.3
// CHECK:STDOUT:   %c: %char = value_binding c, %.loc9_26.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_literal_out_of_bounds.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %str.0a6: %ptr.fb0 = string_literal "Test" [concrete]
// CHECK:STDOUT:   %int_4.2df: %u64 = int_value 4 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.0a6, %int_4.2df) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %IndexWith.type.8ab: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %IndexWith.At.type.1ab: type = fn_type @IndexWith.At, @IndexWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %T.6be: %ImplicitAs.type.d14 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.4f6: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%T.6be) [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.362: %str.as.IndexWith.impl.At.type.4f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3ced: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.462, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3ced) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.4ec: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.1d2: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.17e: %str.as.IndexWith.impl.At.type.1d2 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.8ab = facet_value %str.ee0, (%IndexWith.impl_witness.4ec) [concrete]
// CHECK:STDOUT:   %.d99: type = fn_type_with_self_type %IndexWith.At.type.1ab, %IndexWith.facet [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.bound: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.17e [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %str.as.IndexWith.impl.At.17e, @str.as.IndexWith.impl.At(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.484 = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.14a: @str.as.IndexWith.impl.%str.as.IndexWith.impl.At.type (%str.as.IndexWith.impl.At.type.4f6) = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, loaded [symbolic = @str.as.IndexWith.impl.%str.as.IndexWith.impl.At (constants.%str.as.IndexWith.impl.At.362)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.484, %Core.import_ref.14a), @str.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.8f7 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.462 = impl_witness_table (%Core.import_ref.8f7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLiteralOutOfBounds() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4.loc9_17: %u64 = int_value 4 [concrete = constants.%int_4.2df]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_4.loc9_17) [concrete = constants.%String.val]
// CHECK:STDOUT:   %int_4.loc9_24: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem1: %.d99 = impl_witness_access constants.%IndexWith.impl_witness.4ec, element1 [concrete = constants.%str.as.IndexWith.impl.At.17e]
// CHECK:STDOUT:   %bound_method.loc9_25.1: <bound method> = bound_method %String.val, %impl.elem1 [concrete = constants.%str.as.IndexWith.impl.At.bound]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_25.1: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_25.1: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_25.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc9_25.2: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc9_25.2: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc9_25.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @str.as.IndexWith.impl.At(constants.%ImplicitAs.facet) [concrete = constants.%str.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_25.2: <bound method> = bound_method %String.val, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call: init %char = call %bound_method.loc9_25.2(%String.val, %int_4.loc9_24)
// CHECK:STDOUT:   %.loc9_25.3: %char = value_of_initializer %str.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc9_25.4: %char = converted %str.as.IndexWith.impl.At.call, %.loc9_25.3
// CHECK:STDOUT:   %c: %char = value_binding c, %.loc9_25.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestWrongType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %c: %char = value_binding c, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %IndexWith.type.8ab: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %IndexWith.At.type.1ab: type = fn_type @IndexWith.At, @IndexWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %T.6be: %ImplicitAs.type.d14 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.4f6: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%T.6be) [symbolic]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.362: %str.as.IndexWith.impl.At.type.4f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3ced: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.462, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3ced) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.4ec: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.type.1d2: type = fn_type @str.as.IndexWith.impl.At, @str.as.IndexWith.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.17e: %str.as.IndexWith.impl.At.type.1d2 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.8ab = facet_value %str.ee0, (%IndexWith.impl_witness.4ec) [concrete]
// CHECK:STDOUT:   %.d99: type = fn_type_with_self_type %IndexWith.At.type.1ab, %IndexWith.facet [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.bound: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.17e [concrete]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %str.as.IndexWith.impl.At.17e, @str.as.IndexWith.impl.At(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %String.val, %str.as.IndexWith.impl.At.specific_fn [concrete]
// CHECK:STDOUT:   %int_84: %char = int_value 84 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.484 = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.14a: @str.as.IndexWith.impl.%str.as.IndexWith.impl.At.type (%str.as.IndexWith.impl.At.type.4f6) = import_ref Core//prelude/types/string, loc{{\d+_\d+}}, loaded [symbolic = @str.as.IndexWith.impl.%str.as.IndexWith.impl.At (constants.%str.as.IndexWith.impl.At.362)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.484, %Core.import_ref.14a), @str.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.8f7 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.462 = impl_witness_table (%Core.import_ref.8f7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStringIndexing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str: %ptr.fb0 = string_literal "Test" [concrete = constants.%str.0a6]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_4) [concrete = constants.%String.val]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem1: %.d99 = impl_witness_access constants.%IndexWith.impl_witness.4ec, element1 [concrete = constants.%str.as.IndexWith.impl.At.17e]
// CHECK:STDOUT:   %bound_method.loc5_25.1: <bound method> = bound_method %String.val, %impl.elem1 [concrete = constants.%str.as.IndexWith.impl.At.bound]
// CHECK:STDOUT:   %ImplicitAs.facet.loc5_25.1: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc5_25.1: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc5_25.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc5_25.2: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.3ced) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc5_25.2: %ImplicitAs.type.d14 = converted Core.IntLiteral, %ImplicitAs.facet.loc5_25.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @str.as.IndexWith.impl.At(constants.%ImplicitAs.facet) [concrete = constants.%str.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_25.2: <bound method> = bound_method %String.val, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call: init %char = call %bound_method.loc5_25.2(%String.val, %int_0) [concrete = constants.%int_84]
// CHECK:STDOUT:   %.loc5_25.3: %char = value_of_initializer %str.as.IndexWith.impl.At.call [concrete = constants.%int_84]
// CHECK:STDOUT:   %.loc5_25.4: %char = converted %str.as.IndexWith.impl.At.call, %.loc5_25.3 [concrete = constants.%int_84]
// CHECK:STDOUT:   %c: %char = value_binding c, %.loc5_25.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
