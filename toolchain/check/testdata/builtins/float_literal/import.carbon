// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float_literal/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float_literal/import.carbon

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.FloatLiteral();
}

class C {}

impl C as I where .val = 0.1 {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> Core.FloatLiteral() {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.e01 [concrete]
// CHECK:STDOUT:   %float.3fedf4.2: Core.FloatLiteral = float_literal_value 1e-1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.730: %I.assoc_type = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.dec: Core.FloatLiteral = import_ref Main//symbolic_decl, loc10_30, loaded [concrete = constants.%float.3fedf4.2]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.dec), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.e01: Core.FloatLiteral = import_ref Main//symbolic_decl, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: Core.FloatLiteral = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> Core.FloatLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.730 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0: Core.FloatLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float.3fedf4.2]
// CHECK:STDOUT:   return %impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
