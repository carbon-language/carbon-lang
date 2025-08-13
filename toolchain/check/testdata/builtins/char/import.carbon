// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char/import.carbon

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.Char;
}

class C {}

impl C as I where .val = 'a' {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> Core.Char {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- literal_symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.CharLiteral();
}

class C {}

impl C as I where .val = 'a' {}

// --- literal_import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "literal_symbolic_decl";

fn F() -> Core.CharLiteral() {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_97: %Char = int_value 97 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.941: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.8e8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.da1: %I.assoc_type = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0.941]
// CHECK:STDOUT:   %Main.import_ref.d0c: %Char = import_ref Main//symbolic_decl, loc10_30, loaded [concrete = constants.%int_97]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d0c), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.8e8: %Char = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %Char = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %Char {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.da1 [concrete = constants.%assoc0.941]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: %Char = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_97]
// CHECK:STDOUT:   return %impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.dcc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//literal_symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//literal_symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.696: %I.assoc_type = import_ref Main//literal_symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.43e: Core.CharLiteral = import_ref Main//literal_symbolic_decl, loc10_30, loaded [concrete = constants.%.54f]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.43e), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.dcc: Core.CharLiteral = import_ref Main//literal_symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: Core.CharLiteral = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> Core.CharLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.696 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: Core.CharLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%.54f]
// CHECK:STDOUT:   return %impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
