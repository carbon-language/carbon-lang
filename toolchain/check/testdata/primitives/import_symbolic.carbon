// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/import_symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/import_symbolic.carbon

// --- bool.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! bool;
}

class C {
  impl as I where .value = false {}
}

// --- import_bool.carbon

library "[[@TEST_NAME]]";
import library "bool";

fn F() -> bool {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- char.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! Core.Char;
}

class C {}

impl C as I where .value = 'a' {}

// --- import_char.carbon

library "[[@TEST_NAME]]";
import library "char";

fn F() -> Core.Char {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- char_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! Core.CharLiteral();
}

class C {}

impl C as I where .value = 'a' {}

// --- import_char_literal.carbon

library "[[@TEST_NAME]]";
import library "char_literal";

fn F() -> Core.CharLiteral() {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- float.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! f64;
}

class C {}

impl C as I where .value = 0.1 {}

// --- import_float.carbon

library "[[@TEST_NAME]]";
import library "float";

fn F() -> f64 {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- float_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! Core.FloatLiteral();
}

class C {}

impl C as I where .value = 0.1 {}

// --- import_float_literal.carbon

library "[[@TEST_NAME]]";
import library "float_literal";

fn F() -> Core.FloatLiteral() {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- int.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! i32;
}

class C {}

impl C as I where .value = 8 {}

// --- import_int.carbon

library "[[@TEST_NAME]]";
import library "int";

fn F() -> i32 {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- int_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! Core.IntLiteral();
}

class C {}

impl C as I where .value = 7 {}

// --- import_int_literal.carbon

library "[[@TEST_NAME]]";
import library "int_literal";

fn F() -> Core.IntLiteral() {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// --- uint.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! u32;
}

class C {}

impl C as I where .value = 8 {}

// --- unsigned_uint.carbon

library "[[@TEST_NAME]]";
import library "uint";

fn F() -> u32 {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.ce7: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.fd4 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ba0: <witness> = impl_witness imports.%Copy.impl_witness_table.d8d [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.ba0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.fa0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7ff: type = fn_type_with_self_type %Copy.WithSelf.Op.type.fa0, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//bool, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//bool, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.f87: %I.assoc_type = import_ref Main//bool, loc5_12, loaded [concrete = constants.%assoc0.ce7]
// CHECK:STDOUT:   %Main.import_ref.d18: bool = import_ref Main//bool, loc9_34, loaded [concrete = constants.%false]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d18), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.fd4: bool = import_ref Main//bool, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: bool = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7b4: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d8d = impl_witness_table (%Core.import_ref.7b4), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.f87 [concrete = constants.%assoc0.ce7]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: bool = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.7ff = impl_witness_access constants.%Copy.impl_witness.ba0, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%false]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.31d: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.ddc [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.416: <witness> = impl_witness imports.%Copy.impl_witness_table.cef [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.416) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.981: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.981, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_97, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.296: %I.assoc_type = import_ref Main//char, loc5_12, loaded [concrete = constants.%assoc0.31d]
// CHECK:STDOUT:   %Main.import_ref.68b: %char = import_ref Main//char, loc10_32, loaded [concrete = constants.%int_97]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.68b), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.ddc: %char = import_ref Main//char, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %char = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.c5d: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.cef = impl_witness_table (%Core.import_ref.c5d), @char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %char {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.296 [concrete = constants.%assoc0.31d]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %char = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_97]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.e28 = impl_witness_access constants.%Copy.impl_witness.416, element0 [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%char.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.call: init %char = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%int_97]
// CHECK:STDOUT:   return %char.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.42a: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.799 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.3a1: <witness> = impl_witness imports.%Copy.impl_witness_table.96f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.CharLiteral, (%Copy.impl_witness.3a1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.2cb: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2d9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.2cb, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.type: type = fn_type @Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op: %Core.CharLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %.54f, %Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a15: %I.assoc_type = import_ref Main//char_literal, loc5_12, loaded [concrete = constants.%assoc0.42a]
// CHECK:STDOUT:   %Main.import_ref.43e: Core.CharLiteral = import_ref Main//char_literal, loc10_32, loaded [concrete = constants.%.54f]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.43e), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.799: Core.CharLiteral = import_ref Main//char_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.CharLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7b4: %Core.CharLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.96f = impl_witness_table (%Core.import_ref.7b4), @Core.CharLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: Core.CharLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.a15 [concrete = constants.%assoc0.42a]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.CharLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%.54f]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.2d9 = impl_witness_access constants.%Copy.impl_witness.3a1, element0 [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.call: init Core.CharLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%.54f]
// CHECK:STDOUT:   return %Core.CharLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.9a0: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %float: %f64.9a0 = float_value 0.10000000000000001 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.5d6: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.b26 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.44e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.30e: %Float.as.Copy.impl.Op.type.44e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c2d: <witness> = impl_witness imports.%Copy.impl_witness_table.fe1, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.d57: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.447: %Float.as.Copy.impl.Op.type.d57 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.9a0, (%Copy.impl_witness.c2d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.127: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Copy.WithSelf.Op.type.127, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.447 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.447, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.d20: %I.assoc_type = import_ref Main//float, loc5_12, loaded [concrete = constants.%assoc0.5d6]
// CHECK:STDOUT:   %Main.import_ref.d12: %f64.9a0 = import_ref Main//float, loc10_32, loaded [concrete = constants.%float]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d12), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b26: %f64.9a0 = import_ref Main//float, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %f64.9a0 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4b3: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.44e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.30e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fe1 = impl_witness_table (%Core.import_ref.4b3), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %f64.9a0 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.d20 [concrete = constants.%assoc0.5d6]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %f64.9a0 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.2ac = impl_witness_access constants.%Copy.impl_witness.c2d, element0 [concrete = constants.%Float.as.Copy.impl.Op.447]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.9a0 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%float]
// CHECK:STDOUT:   return %Float.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.038: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.4db [concrete]
// CHECK:STDOUT:   %float.d5ab3d.2: Core.FloatLiteral = float_literal_value 1e-1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a41: <witness> = impl_witness imports.%Copy.impl_witness_table.f81 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.FloatLiteral, (%Copy.impl_witness.a41) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7fc: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.061: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7fc, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.type: type = fn_type @Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op: %Core.FloatLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %float.d5ab3d.2, %Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.bce: %I.assoc_type = import_ref Main//float_literal, loc5_12, loaded [concrete = constants.%assoc0.038]
// CHECK:STDOUT:   %Main.import_ref.78a: Core.FloatLiteral = import_ref Main//float_literal, loc10_32, loaded [concrete = constants.%float.d5ab3d.2]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.78a), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.4db: Core.FloatLiteral = import_ref Main//float_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.FloatLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7b4: %Core.FloatLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.f81 = impl_witness_table (%Core.import_ref.7b4), @Core.FloatLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: Core.FloatLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.bce [concrete = constants.%assoc0.038]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.FloatLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float.d5ab3d.2]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.061 = impl_witness_access constants.%Copy.impl_witness.a41, element0 [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.call: init Core.FloatLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%float.d5ab3d.2]
// CHECK:STDOUT:   return %Core.FloatLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.d74: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.048 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ab4: <witness> = impl_witness imports.%Copy.impl_witness_table.193, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.ab4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e27: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.98b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e27, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.c4a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3ef: %I.assoc_type = import_ref Main//int, loc5_12, loaded [concrete = constants.%assoc0.d74]
// CHECK:STDOUT:   %Main.import_ref.fc6: %i32 = import_ref Main//int, loc10_30, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.fc6), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.048: %i32 = import_ref Main//int, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %i32 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d9d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.193 = impl_witness_table (%Core.import_ref.d9d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.3ef [concrete = constants.%assoc0.d74]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %i32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.98b = impl_witness_access constants.%Copy.impl_witness.ab4, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%int_8]
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.93c: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.721 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6df: <witness> = impl_witness imports.%Copy.impl_witness_table.ab7 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.6df) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ee4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.281: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ee4, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_7, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.d8c: %I.assoc_type = import_ref Main//int_literal, loc5_12, loaded [concrete = constants.%assoc0.93c]
// CHECK:STDOUT:   %Main.import_ref.bc3: Core.IntLiteral = import_ref Main//int_literal, loc10_30, loaded [concrete = constants.%int_7]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.bc3), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.721: Core.IntLiteral = import_ref Main//int_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.IntLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7b4: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ab7 = impl_witness_table (%Core.import_ref.7b4), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.d8c [concrete = constants.%assoc0.93c]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.IntLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_7]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.281 = impl_witness_access constants.%Copy.impl_witness.6df, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%int_7]
// CHECK:STDOUT:   return %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_uint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %u32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.87c: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.a0d [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.a58: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.dc5: %UInt.as.Copy.impl.Op.type.a58 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ff6: <witness> = impl_witness imports.%Copy.impl_witness_table.54d, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.3cf: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.6ad: %UInt.as.Copy.impl.Op.type.3cf = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u32, (%Copy.impl_witness.ff6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5c6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.0e4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5c6, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.6ad [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.6ad, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//uint, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//uint, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.1ad: %I.assoc_type = import_ref Main//uint, loc5_12, loaded [concrete = constants.%assoc0.87c]
// CHECK:STDOUT:   %Main.import_ref.250: %u32 = import_ref Main//uint, loc10_30, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.250), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a0d: %u32 = import_ref Main//uint, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %u32 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.047: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.a58) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.dc5)]
// CHECK:STDOUT:   %Copy.impl_witness_table.54d = impl_witness_table (%Core.import_ref.047), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.1ad [concrete = constants.%assoc0.87c]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %u32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.0e4 = impl_witness_access constants.%Copy.impl_witness.ff6, element0 [concrete = constants.%UInt.as.Copy.impl.Op.6ad]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u32 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%int_8]
// CHECK:STDOUT:   return %UInt.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
