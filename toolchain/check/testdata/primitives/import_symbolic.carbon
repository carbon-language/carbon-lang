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
// CHECK:STDOUT:   %assoc0.96e: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.073 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.de3: <witness> = impl_witness imports.%Copy.impl_witness_table.34b [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.de3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.079: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.022: type = fn_type_with_self_type %Copy.WithSelf.Op.type.079, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//bool, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//bool, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a82: %I.assoc_type = import_ref Main//bool, loc5_12, loaded [concrete = constants.%assoc0.96e]
// CHECK:STDOUT:   %Main.import_ref.d18: bool = import_ref Main//bool, loc9_34, loaded [concrete = constants.%false]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d18), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.073: bool = import_ref Main//bool, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: bool = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.34b = impl_witness_table (%Core.import_ref.f63), @bool.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.a82 [concrete = constants.%assoc0.96e]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: bool = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.022 = impl_witness_access constants.%Copy.impl_witness.de3, element0 [concrete = constants.%bool.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %assoc0.25c: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.676 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.937: <witness> = impl_witness imports.%Copy.impl_witness_table.045 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.937) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.386: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e5e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.386, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_97, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.740: %I.assoc_type = import_ref Main//char, loc5_12, loaded [concrete = constants.%assoc0.25c]
// CHECK:STDOUT:   %Main.import_ref.118: %char = import_ref Main//char, loc10_32, loaded [concrete = constants.%int_97]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.118), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.676: %char = import_ref Main//char, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %char = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.045 = impl_witness_table (%Core.import_ref.f63), @char.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.740 [concrete = constants.%assoc0.25c]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %char = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_97]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.e5e = impl_witness_access constants.%Copy.impl_witness.937, element0 [concrete = constants.%char.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %assoc0.ece: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.aaf [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f9b: <witness> = impl_witness imports.%Copy.impl_witness_table.08c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.CharLiteral, (%Copy.impl_witness.f9b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bf4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.840: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bf4, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.type: type = fn_type @Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op: %Core.CharLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %.54f, %Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.421: %I.assoc_type = import_ref Main//char_literal, loc5_12, loaded [concrete = constants.%assoc0.ece]
// CHECK:STDOUT:   %Main.import_ref.43e: Core.CharLiteral = import_ref Main//char_literal, loc10_32, loaded [concrete = constants.%.54f]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.43e), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.aaf: Core.CharLiteral = import_ref Main//char_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.CharLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63: %Core.CharLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.08c = impl_witness_table (%Core.import_ref.f63), @Core.CharLiteral.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.421 [concrete = constants.%assoc0.ece]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.CharLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%.54f]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.840 = impl_witness_access constants.%Copy.impl_witness.f9b, element0 [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %float: %f64.b55 = float_value 0.10000000000000001 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.26f: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.1f2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.125: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bfc: %Float.as.Copy.impl.Op.type.125 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.811: <witness> = impl_witness imports.%Copy.impl_witness_table.384, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.7a5: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bcc: %Float.as.Copy.impl.Op.type.7a5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.b55, (%Copy.impl_witness.811) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.286: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.c7d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.286, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.bcc [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.bcc, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.276: %I.assoc_type = import_ref Main//float, loc5_12, loaded [concrete = constants.%assoc0.26f]
// CHECK:STDOUT:   %Main.import_ref.6ea: %f64.b55 = import_ref Main//float, loc10_32, loaded [concrete = constants.%float]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.6ea), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.1f2: %f64.b55 = import_ref Main//float, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %f64.b55 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.461: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.125) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.bfc)]
// CHECK:STDOUT:   %Copy.impl_witness_table.384 = impl_witness_table (%Core.import_ref.461), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %f64.b55 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.276 [concrete = constants.%assoc0.26f]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %f64.b55 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.c7d = impl_witness_access constants.%Copy.impl_witness.811, element0 [concrete = constants.%Float.as.Copy.impl.Op.bcc]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.b55 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%float]
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
// CHECK:STDOUT:   %assoc0.518: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.08a [concrete]
// CHECK:STDOUT:   %float.d5ab3d.2: Core.FloatLiteral = float_literal_value 1e-1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.686: <witness> = impl_witness imports.%Copy.impl_witness_table.d0c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.FloatLiteral, (%Copy.impl_witness.686) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.2a0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.90a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.2a0, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.type: type = fn_type @Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op: %Core.FloatLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %float.d5ab3d.2, %Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5bd: %I.assoc_type = import_ref Main//float_literal, loc5_12, loaded [concrete = constants.%assoc0.518]
// CHECK:STDOUT:   %Main.import_ref.78a: Core.FloatLiteral = import_ref Main//float_literal, loc10_32, loaded [concrete = constants.%float.d5ab3d.2]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.78a), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.08a: Core.FloatLiteral = import_ref Main//float_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.FloatLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63: %Core.FloatLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.d0c = impl_witness_table (%Core.import_ref.f63), @Core.FloatLiteral.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.5bd [concrete = constants.%assoc0.518]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.FloatLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float.d5ab3d.2]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.90a = impl_witness_access constants.%Copy.impl_witness.686, element0 [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %assoc0.cb5: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.102 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.ce7 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.c2e: %I.assoc_type = import_ref Main//int, loc5_12, loaded [concrete = constants.%assoc0.cb5]
// CHECK:STDOUT:   %Main.import_ref.4ce: %i32 = import_ref Main//int, loc10_30, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.4ce), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.102: %i32 = import_ref Main//int, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %i32 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.c2e [concrete = constants.%assoc0.cb5]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %i32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
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
// CHECK:STDOUT:   %assoc0.b97: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.ceb [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.969: <witness> = impl_witness imports.%Copy.impl_witness_table.365 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.969) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.535: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b58: type = fn_type_with_self_type %Copy.WithSelf.Op.type.535, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_7, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.265: %I.assoc_type = import_ref Main//int_literal, loc5_12, loaded [concrete = constants.%assoc0.b97]
// CHECK:STDOUT:   %Main.import_ref.bc3: Core.IntLiteral = import_ref Main//int_literal, loc10_30, loaded [concrete = constants.%int_7]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.bc3), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.ceb: Core.IntLiteral = import_ref Main//int_literal, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.IntLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.365 = impl_witness_table (%Core.import_ref.f63), @Core.IntLiteral.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.265 [concrete = constants.%assoc0.b97]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.IntLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_7]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.b58 = impl_witness_access constants.%Copy.impl_witness.969, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %assoc0.0ce: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.078 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.743: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.521: %UInt.as.Copy.impl.Op.type.743 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.cfa: <witness> = impl_witness imports.%Copy.impl_witness_table.306, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.46f: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.a73: %UInt.as.Copy.impl.Op.type.46f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u32, (%Copy.impl_witness.cfa) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7b4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7b4, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.a73 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.a73, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//uint, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//uint, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a42: %I.assoc_type = import_ref Main//uint, loc5_12, loaded [concrete = constants.%assoc0.0ce]
// CHECK:STDOUT:   %Main.import_ref.0b5: %u32 = import_ref Main//uint, loc10_30, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.0b5), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.078: %u32 = import_ref Main//uint, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %u32 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.3cf: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.743) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.521)]
// CHECK:STDOUT:   %Copy.impl_witness_table.306 = impl_witness_table (%Core.import_ref.3cf), @UInt.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.a42 [concrete = constants.%assoc0.0ce]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %u32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.637 = impl_witness_access constants.%Copy.impl_witness.cfa, element0 [concrete = constants.%UInt.as.Copy.impl.Op.a73]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u32 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%int_8]
// CHECK:STDOUT:   return %UInt.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
