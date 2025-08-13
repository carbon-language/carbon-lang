// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/import.carbon

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! i32;
}

class C {}

impl C as I where .val = 8 {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> i32 {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- unsigned_symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! u32;
}

class C {}

impl C as I where .val = 8 {}

// --- unsigned_import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "unsigned_symbolic_decl";

fn F() -> u32 {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.aac: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.05e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.ba4: %I.assoc_type = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0.aac]
// CHECK:STDOUT:   %Main.import_ref.330: %i32 = import_ref Main//symbolic_decl, loc10_28, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.330), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.05e: %i32 = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %i32 = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.ba4 [concrete = constants.%assoc0.aac]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: %i32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   return %impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %u32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.633: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.830 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//unsigned_symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//unsigned_symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.d18: %I.assoc_type = import_ref Main//unsigned_symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0.633]
// CHECK:STDOUT:   %Main.import_ref.647: %u32 = import_ref Main//unsigned_symbolic_decl, loc10_28, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.647), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.830: %u32 = import_ref Main//unsigned_symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %u32 = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.d18 [concrete = constants.%assoc0.633]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: %u32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   return %impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
