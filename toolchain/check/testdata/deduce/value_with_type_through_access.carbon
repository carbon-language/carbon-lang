// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/value_with_type_through_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/value_with_type_through_access.carbon

// --- tuple_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! (type, )) {}

// `a` is received as a value, and has a type that requires going through
// TupleAccess. Building the value representation must handle this indirection.
fn F[T:! (type, )](unused x: HoldsType(T), unused a: T.0) {}

class C {}

fn G() {
  F({} as HoldsType((C, )), {});
}

// --- struct_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! {.t: type}) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
fn F[T:! {.t: type}](unused x: HoldsType(T), unused a: T.t) {}

class C {}

fn G() {
  F({} as HoldsType({.t = C}), {});
}

// --- fail_todo_class_access.carbon

library "[[@TEST_NAME]]";

class Class {
  var t: type;
}

class HoldsType(T:! Class) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
//
// TODO: The `Class` value in `T` is a comptime constant and this expression
// should be able to eval to a type. We would expect the `.` in `T.t` to be a
// ClassElementAccess but it is an AcquireValue, which has a runtime value.
//
// CHECK:STDERR: fail_todo_class_access.carbon:[[@LINE+4]]:51: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T:! Class](unused x: HoldsType(T), unused a: T.t) {}
// CHECK:STDERR:                                                   ^~~
// CHECK:STDERR:
fn F[T:! Class](unused x: HoldsType(T), unused a: T.t) {}

class C {}

fn G(c:! Class) {
  F({} as HoldsType(c), {});
}

fn H() {
  G({.t = C});
}

// --- fail_todo_array_index.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! array(type, 1)) {}

// `a` is received as a value, and has a type that requires going through
// ArrayIndex. Building the value representation must handle this indirection.
// CHECK:STDERR: fail_todo_array_index.carbon:[[@LINE+4]]:60: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T:! array(type, 1)](unused x: HoldsType(T), unused a: T[0]) {}
// CHECK:STDERR:                                                            ^~~~
// CHECK:STDERR:
fn F[T:! array(type, 1)](unused x: HoldsType(T), unused a: T[0]) {}

class C {}

fn G() {
  F({} as HoldsType((C, ) as array(type, 1)), {});
}



// CHECK:STDOUT: --- tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.0d2: %tuple.type = tuple_value (type) [concrete]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.a83: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.b2f: type = pattern_type %HoldsType.a83 [symbolic]
// CHECK:STDOUT:   %x.param_patt.321: %pattern_type.b2f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a80: %pattern_type.b2f = at_binding_pattern x, %x.param_patt.321 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %a.param_patt.d7c: %pattern_type.e66 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.803: %pattern_type.e66 = at_binding_pattern a, %a.param_patt.d7c [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9de: <witness> = require_complete_type %HoldsType.a83 [symbolic]
// CHECK:STDOUT:   %require_complete.e7a: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.e45: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.30f: type = class_type @HoldsType, @HoldsType(%tuple.e45) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.30f = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ff4: type = pattern_type %HoldsType.30f [concrete]
// CHECK:STDOUT:   %x.param_patt.995: %pattern_type.ff4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.ea0: %pattern_type.ff4 = at_binding_pattern x, %x.param_patt.995 [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.159: %pattern_type.98b = at_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.e45) [concrete]
// CHECK:STDOUT:   %.eb0: ref %HoldsType.30f = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc13_30.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.72d: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.bc7: <bound method> = bound_method %.eb0, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_22: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc4_28.2: %tuple.type = tuple_literal (%.loc4_22) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc4_28.3: type = converted %.loc4_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_28.1: @F.%pattern_type.loc8_28 (%pattern_type.b2f) = value_param_pattern [symbolic = %x.param_patt.loc8_28.2 (constants.%x.param_patt.321)]
// CHECK:STDOUT:     %x.patt.loc8_28.1: @F.%pattern_type.loc8_28 (%pattern_type.b2f) = at_binding_pattern x, %x.param_patt.loc8_28.1 [symbolic = %x.patt.loc8_28.2 (constants.%x.patt.a80)]
// CHECK:STDOUT:     %a.param_patt.loc8_52.1: @F.%pattern_type.loc8_52 (%pattern_type.e66) = value_param_pattern [symbolic = %a.param_patt.loc8_52.2 (constants.%a.param_patt.d7c)]
// CHECK:STDOUT:     %a.patt.loc8_52.1: @F.%pattern_type.loc8_52 (%pattern_type.e66) = at_binding_pattern a, %a.param_patt.loc8_52.1 [symbolic = %a.patt.loc8_52.2 (constants.%a.patt.803)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_11: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc8_17.2: %tuple.type = tuple_literal (%.loc8_11) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc8_17.3: type = converted %.loc8_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.a83) = value_param call_param0
// CHECK:STDOUT:     %.loc8_41: type = splice_block %HoldsType.loc8_41.2 [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.a83)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_40: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_41.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.a83)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_41.1 (%HoldsType.a83) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_55: type = splice_block %tuple.elem0.loc8_55.2 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_54: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc8_55.2: type = tuple_access %T.ref.loc8_54, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %tuple.type) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.a83
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %tuple.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.f1e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_41.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.a83)]
// CHECK:STDOUT:   %pattern_type.loc8_28: type = pattern_type %HoldsType.loc8_41.1 [symbolic = %pattern_type.loc8_28 (constants.%pattern_type.b2f)]
// CHECK:STDOUT:   %x.param_patt.loc8_28.2: @F.%pattern_type.loc8_28 (%pattern_type.b2f) = value_param_pattern [symbolic = %x.param_patt.loc8_28.2 (constants.%x.param_patt.321)]
// CHECK:STDOUT:   %x.patt.loc8_28.2: @F.%pattern_type.loc8_28 (%pattern_type.b2f) = at_binding_pattern x, %x.param_patt.loc8_28.2 [symbolic = %x.patt.loc8_28.2 (constants.%x.patt.a80)]
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1: type = tuple_access %T.loc8_7.1, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_52: type = pattern_type %tuple.elem0.loc8_55.1 [symbolic = %pattern_type.loc8_52 (constants.%pattern_type.e66)]
// CHECK:STDOUT:   %a.param_patt.loc8_52.2: @F.%pattern_type.loc8_52 (%pattern_type.e66) = value_param_pattern [symbolic = %a.param_patt.loc8_52.2 (constants.%a.param_patt.d7c)]
// CHECK:STDOUT:   %a.patt.loc8_52.2: @F.%pattern_type.loc8_52 (%pattern_type.e66) = at_binding_pattern a, %a.param_patt.loc8_52.2 [symbolic = %a.patt.loc8_52.2 (constants.%a.patt.803)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_28: <witness> = require_complete_type %HoldsType.loc8_41.1 [symbolic = %require_complete.loc8_28 (constants.%require_complete.9de)]
// CHECK:STDOUT:   %require_complete.loc8_52: <witness> = require_complete_type %tuple.elem0.loc8_55.1 [symbolic = %require_complete.loc8_52 (constants.%require_complete.e7a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.a83), %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc13_25: %tuple.type = tuple_literal (%C.ref) [concrete = constants.%tuple.e45]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple.e45]
// CHECK:STDOUT:   %.loc13_26: %tuple.type = converted %.loc13_25, %tuple [concrete = constants.%tuple.e45]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.e45) [concrete = constants.%HoldsType.30f]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.30f = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.30f to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.30f = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.e45) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.30f = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.eb0]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.30f = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C to %.loc13_30.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_30.1, %.loc13_30.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.5: ref %C = temporary %.loc13_30.2, %.loc13_30.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_30.6: %C = acquire_value %.loc13_30.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_30.6)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_30: init %empty_tuple.type = call constants.%Destroy.Op.bound.72d(constants.%.115)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.Op.bound.bc7(constants.%.eb0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_30.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_30.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_8(%self.param: ref %HoldsType.30f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.a83
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.b2f
// CHECK:STDOUT:   %x.param_patt.loc8_28.2 => constants.%x.param_patt.321
// CHECK:STDOUT:   %x.patt.loc8_28.2 => constants.%x.patt.a80
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_52 => constants.%pattern_type.e66
// CHECK:STDOUT:   %a.param_patt.loc8_52.2 => constants.%a.param_patt.d7c
// CHECK:STDOUT:   %a.patt.loc8_52.2 => constants.%a.patt.803
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.e45) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%tuple.e45
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.e45) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%tuple.e45
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.30f
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.ff4
// CHECK:STDOUT:   %x.param_patt.loc8_28.2 => constants.%x.param_patt.995
// CHECK:STDOUT:   %x.patt.loc8_28.2 => constants.%x.patt.ea0
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_52 => constants.%pattern_type.98b
// CHECK:STDOUT:   %a.param_patt.loc8_52.2 => constants.%a.param_patt.c88
// CHECK:STDOUT:   %a.patt.loc8_52.2 => constants.%a.patt.159
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_28 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc8_52 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %pattern_type.7f2: type = pattern_type %struct_type.t [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %struct_type.t = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.07e: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.ce7: type = pattern_type %HoldsType.07e [symbolic]
// CHECK:STDOUT:   %x.param_patt.ade: %pattern_type.ce7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.d2a: %pattern_type.ce7 = at_binding_pattern x, %x.param_patt.ade [symbolic]
// CHECK:STDOUT:   %.424: type = struct_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.92c: type = pattern_type %.424 [symbolic]
// CHECK:STDOUT:   %a.param_patt.314: %pattern_type.92c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.92f: %pattern_type.92c = at_binding_pattern a, %a.param_patt.314 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d40: <witness> = require_complete_type %HoldsType.07e [symbolic]
// CHECK:STDOUT:   %require_complete.ba0: <witness> = require_complete_type %.424 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.53c: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.53c = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1fc: type = pattern_type %HoldsType.53c [concrete]
// CHECK:STDOUT:   %x.param_patt.4a0: %pattern_type.1fc = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.5e6: %pattern_type.1fc = at_binding_pattern x, %x.param_patt.4a0 [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.159: %pattern_type.98b = at_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %.344: ref %HoldsType.53c = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc13_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.72d: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.e30: <bound method> = bound_method %.344, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_30: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_26: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_30.1: @F.%pattern_type.loc8_30 (%pattern_type.ce7) = value_param_pattern [symbolic = %x.param_patt.loc8_30.2 (constants.%x.param_patt.ade)]
// CHECK:STDOUT:     %x.patt.loc8_30.1: @F.%pattern_type.loc8_30 (%pattern_type.ce7) = at_binding_pattern x, %x.param_patt.loc8_30.1 [symbolic = %x.patt.loc8_30.2 (constants.%x.patt.d2a)]
// CHECK:STDOUT:     %a.param_patt.loc8_54.1: @F.%pattern_type.loc8_54 (%pattern_type.92c) = value_param_pattern [symbolic = %a.param_patt.loc8_54.2 (constants.%a.param_patt.314)]
// CHECK:STDOUT:     %a.patt.loc8_54.1: @F.%pattern_type.loc8_54 (%pattern_type.92c) = at_binding_pattern a, %a.param_patt.loc8_54.1 [symbolic = %a.patt.loc8_54.2 (constants.%a.patt.92f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_15: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.07e) = value_param call_param0
// CHECK:STDOUT:     %.loc8_43: type = splice_block %HoldsType.loc8_43.2 [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.07e)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_43.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.07e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_43.1 (%HoldsType.07e) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_57.1 (%.424) = value_param call_param1
// CHECK:STDOUT:     %.loc8_57.2: type = splice_block %.loc8_57.3 [symbolic = %.loc8_57.1 (constants.%.424)] {
// CHECK:STDOUT:       %T.ref.loc8_56: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_57.3: type = struct_access %T.ref.loc8_56, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_57.1 (%.424) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %struct_type.t) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_18.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.07e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %struct_type.t) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_43.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.07e)]
// CHECK:STDOUT:   %pattern_type.loc8_30: type = pattern_type %HoldsType.loc8_43.1 [symbolic = %pattern_type.loc8_30 (constants.%pattern_type.ce7)]
// CHECK:STDOUT:   %x.param_patt.loc8_30.2: @F.%pattern_type.loc8_30 (%pattern_type.ce7) = value_param_pattern [symbolic = %x.param_patt.loc8_30.2 (constants.%x.param_patt.ade)]
// CHECK:STDOUT:   %x.patt.loc8_30.2: @F.%pattern_type.loc8_30 (%pattern_type.ce7) = at_binding_pattern x, %x.param_patt.loc8_30.2 [symbolic = %x.patt.loc8_30.2 (constants.%x.patt.d2a)]
// CHECK:STDOUT:   %.loc8_57.1: type = struct_access %T.loc8_7.1, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:   %pattern_type.loc8_54: type = pattern_type %.loc8_57.1 [symbolic = %pattern_type.loc8_54 (constants.%pattern_type.92c)]
// CHECK:STDOUT:   %a.param_patt.loc8_54.2: @F.%pattern_type.loc8_54 (%pattern_type.92c) = value_param_pattern [symbolic = %a.param_patt.loc8_54.2 (constants.%a.param_patt.314)]
// CHECK:STDOUT:   %a.patt.loc8_54.2: @F.%pattern_type.loc8_54 (%pattern_type.92c) = at_binding_pattern a, %a.param_patt.loc8_54.2 [symbolic = %a.patt.loc8_54.2 (constants.%a.patt.92f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_30: <witness> = require_complete_type %HoldsType.loc8_43.1 [symbolic = %require_complete.loc8_30 (constants.%require_complete.d40)]
// CHECK:STDOUT:   %require_complete.loc8_54: <witness> = require_complete_type %.loc8_57.1 [symbolic = %require_complete.loc8_54 (constants.%require_complete.ba0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.07e), %a.param: @F.%.loc8_57.1 (%.424)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc13_28: %struct_type.t = struct_literal (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc13_29: %struct_type.t = converted %.loc13_28, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.53c]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.53c = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.53c to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.53c = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.53c = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.344]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.53c = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C to %.loc13_33.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: init %C = converted %.loc13_33.1, %.loc13_33.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.5: ref %C = temporary %.loc13_33.2, %.loc13_33.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_33.6: %C = acquire_value %.loc13_33.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_33.6)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_33: init %empty_tuple.type = call constants.%Destroy.Op.bound.72d(constants.%.115)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.Op.bound.e30(constants.%.344)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_33.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_33.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_8(%self.param: ref %HoldsType.53c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.07e
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.ce7
// CHECK:STDOUT:   %x.param_patt.loc8_30.2 => constants.%x.param_patt.ade
// CHECK:STDOUT:   %x.patt.loc8_30.2 => constants.%x.patt.d2a
// CHECK:STDOUT:   %.loc8_57.1 => constants.%.424
// CHECK:STDOUT:   %pattern_type.loc8_54 => constants.%pattern_type.92c
// CHECK:STDOUT:   %a.param_patt.loc8_54.2 => constants.%a.param_patt.314
// CHECK:STDOUT:   %a.patt.loc8_54.2 => constants.%a.patt.92f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%struct) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%struct
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.53c
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.1fc
// CHECK:STDOUT:   %x.param_patt.loc8_30.2 => constants.%x.param_patt.4a0
// CHECK:STDOUT:   %x.patt.loc8_30.2 => constants.%x.patt.5e6
// CHECK:STDOUT:   %.loc8_57.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_54 => constants.%pattern_type.98b
// CHECK:STDOUT:   %a.param_patt.loc8_54.2 => constants.%a.param_patt.c88
// CHECK:STDOUT:   %a.patt.loc8_54.2 => constants.%a.patt.159
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_30 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc8_54 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_class_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, type [concrete]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %complete_type.509: <witness> = complete_type_witness %struct_type.t [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %T.patt.125: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f25: %Class = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.3500da.1: type = class_type @HoldsType, @HoldsType(%T.f25) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e53: type = pattern_type %HoldsType.3500da.1 [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.e53 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.e53 = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %.9dd: ref type = class_element_access %T.f25, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f4d7d.1: <witness> = require_complete_type %HoldsType.3500da.1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic]
// CHECK:STDOUT:   %c: %Class = symbolic_binding c, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.3500da.2: type = class_type @HoldsType, @HoldsType(%c) [symbolic]
// CHECK:STDOUT:   %require_complete.6f4d7d.2: <witness> = require_complete_type %HoldsType.3500da.2 [symbolic]
// CHECK:STDOUT:   %HoldsType.val.7f1: %HoldsType.3500da.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.638: <witness> = impl_witness imports.%Copy.impl_witness_table.4d6 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.638) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.8b3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.524: type = fn_type_with_self_type %Copy.WithSelf.Op.type.8b3, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %.0de: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%Class.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc30_13.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.0de, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT:   %HoldsType.af5: type = class_type @HoldsType, @HoldsType(%Class.val) [concrete]
// CHECK:STDOUT:   %HoldsType.val.8bc: %HoldsType.af5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.4d6 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt.loc8_18.1: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt.125)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_18.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_18.1 (constants.%T.f25)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc21_7.1: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_7.2 (constants.%T.patt.125)]
// CHECK:STDOUT:     %x.param_patt.loc21_25.1: @F.%pattern_type (%pattern_type.e53) = value_param_pattern [symbolic = %x.param_patt.loc21_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc21_25.1: @F.%pattern_type (%pattern_type.e53) = at_binding_pattern x, %x.param_patt.loc21_25.1 [symbolic = %x.patt.loc21_25.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = at_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_10: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_7.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.3500da.1) = value_param call_param0
// CHECK:STDOUT:     %.loc21_38: type = splice_block %HoldsType.loc21_38.2 [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.3500da.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc21_37: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %HoldsType.loc21_38.2: type = class_type @HoldsType, @HoldsType(constants.%T.f25) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.3500da.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc21_38.1 (%HoldsType.3500da.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc21_51: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:       %.loc21_52.2: ref type = class_element_access %T.ref.loc21_51, element0 [symbolic = %.loc21_52.1 (constants.%.9dd)]
// CHECK:STDOUT:       %.loc21_52.3: type = acquire_value %.loc21_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt.loc25_7.1: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc25_7.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc25_7.2: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_7.1 (constants.%c)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc5: %Class.elem = field_decl t, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.t [concrete = constants.%complete_type.509]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .t = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc8_18.2: %Class) {
// CHECK:STDOUT:   %T.patt.loc8_18.2: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_18.2 (constants.%T.patt.125)]
// CHECK:STDOUT:   %T.loc8_18.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_18.1 (constants.%T.f25)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.3500da.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc21_7.2: %Class) {
// CHECK:STDOUT:   %T.patt.loc21_7.2: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_7.2 (constants.%T.patt.125)]
// CHECK:STDOUT:   %T.loc21_7.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:   %HoldsType.loc21_38.1: type = class_type @HoldsType, @HoldsType(%T.loc21_7.1) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.3500da.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc21_38.1 [symbolic = %pattern_type (constants.%pattern_type.e53)]
// CHECK:STDOUT:   %x.param_patt.loc21_25.2: @F.%pattern_type (%pattern_type.e53) = value_param_pattern [symbolic = %x.param_patt.loc21_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc21_25.2: @F.%pattern_type (%pattern_type.e53) = at_binding_pattern x, %x.param_patt.loc21_25.2 [symbolic = %x.patt.loc21_25.2 (constants.%x.patt)]
// CHECK:STDOUT:   %.loc21_52.1: ref type = class_element_access %T.loc21_7.1, element0 [symbolic = %.loc21_52.1 (constants.%.9dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc21_38.1 [symbolic = %require_complete (constants.%require_complete.6f4d7d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.3500da.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%c.loc25_7.2: %Class) {
// CHECK:STDOUT:   %c.patt.loc25_7.2: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc25_7.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.loc25_7.1: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_7.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2: type = class_type @HoldsType, @HoldsType(%c.loc25_7.1) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.3500da.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc26_22.2 [symbolic = %require_complete (constants.%require_complete.6f4d7d.2)]
// CHECK:STDOUT:   %HoldsType.val: @G.%HoldsType.loc26_22.2 (%HoldsType.3500da.2) = struct_value () [symbolic = %HoldsType.val (constants.%HoldsType.val.7f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc26_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:     %c.ref: %Class = name_ref c, %c.loc25_7.2 [symbolic = %c.loc25_7.1 (constants.%c)]
// CHECK:STDOUT:     %HoldsType.loc26_22.1: type = class_type @HoldsType, @HoldsType(constants.%c) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.3500da.2)]
// CHECK:STDOUT:     %.loc26_6.2: ref @G.%HoldsType.loc26_22.2 (%HoldsType.3500da.2) = temporary_storage
// CHECK:STDOUT:     %.loc26_6.3: init @G.%HoldsType.loc26_22.2 (%HoldsType.3500da.2) to %.loc26_6.2 = class_init () [symbolic = %HoldsType.val (constants.%HoldsType.val.7f1)]
// CHECK:STDOUT:     %.loc26_8: init @G.%HoldsType.loc26_22.2 (%HoldsType.3500da.2) = converted %.loc26_6.1, %.loc26_6.3 [symbolic = %HoldsType.val (constants.%HoldsType.val.7f1)]
// CHECK:STDOUT:     %.loc26_26: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.1: %struct_type.t = struct_literal (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.524 = impl_witness_access constants.%Copy.impl_witness.638, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc30_12.3: ref type = class_element_access %.loc30_12.2, element0
// CHECK:STDOUT:   %.loc30_12.4: init type to %.loc30_12.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.5: init %Class to %.loc30_12.2 = class_init (%.loc30_12.4) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc30_13.1: init %Class = converted %.loc30_12.1, %.loc30_12.5 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc30_13.2: ref %Class = temporary %.loc30_12.2, %.loc30_13.1 [concrete = constants.%.0de]
// CHECK:STDOUT:   %.loc30_13.3: %Class = acquire_value %.loc30_13.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%Class.val) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.0de)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc30_13.1(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc30_13.2(%self.param: ref %struct_type.t) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc30_13.3(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.f25) {
// CHECK:STDOUT:   %T.patt.loc8_18.2 => constants.%T.patt.125
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%T.f25
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f25) {
// CHECK:STDOUT:   %T.patt.loc21_7.2 => constants.%T.patt.125
// CHECK:STDOUT:   %T.loc21_7.1 => constants.%T.f25
// CHECK:STDOUT:   %HoldsType.loc21_38.1 => constants.%HoldsType.3500da.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e53
// CHECK:STDOUT:   %x.param_patt.loc21_25.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc21_25.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc21_52.1 => constants.%.9dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc25_7.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc25_7.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%c) {
// CHECK:STDOUT:   %T.patt.loc8_18.2 => constants.%T.patt.125
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Class.val) {
// CHECK:STDOUT:   %c.patt.loc25_7.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc25_7.1 => constants.%Class.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2 => constants.%HoldsType.af5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %HoldsType.val => constants.%HoldsType.val.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%Class.val) {
// CHECK:STDOUT:   %T.patt.loc8_18.2 => constants.%T.patt.125
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%Class.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, type [concrete]
// CHECK:STDOUT:   %pattern_type.dcb: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %T.patt.4c7: %pattern_type.dcb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.9b7: %array_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.881: type = class_type @HoldsType, @HoldsType(%T.9b7) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.d35: type = pattern_type %HoldsType.881 [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d35 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.d35 = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.80d: <witness> = require_complete_type %HoldsType.881 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.e45: %tuple.type.85c = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.638: <witness> = impl_witness imports.%Copy.impl_witness_table.4d6 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.638) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.8b3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.524: type = fn_type_with_self_type %Copy.WithSelf.Op.type.8b3, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %.1c6: ref %array_type = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %HoldsType.f11: type = class_type @HoldsType, @HoldsType(%array) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc17_27.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.1c6, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.4d6 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.dcb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.4c7)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_34: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc4_27 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.9b7)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_7.1: %pattern_type.dcb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.4c7)]
// CHECK:STDOUT:     %x.param_patt.loc12_34.1: @F.%pattern_type (%pattern_type.d35) = value_param_pattern [symbolic = %x.param_patt.loc12_34.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc12_34.1: @F.%pattern_type (%pattern_type.d35) = at_binding_pattern x, %x.param_patt.loc12_34.1 [symbolic = %x.patt.loc12_34.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = at_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_16: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc12_16 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc12_47.1 (%HoldsType.881) = value_param call_param0
// CHECK:STDOUT:     %.loc12_47: type = splice_block %HoldsType.loc12_47.2 [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.881)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc12_46: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %HoldsType.loc12_47.2: type = class_type @HoldsType, @HoldsType(constants.%T.9b7) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.881)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc12_47.1 (%HoldsType.881) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc12_60: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.loc12_62.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_62.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_62.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_62.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_62.2: %i32 = converted %int_0, %.loc12_62.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_63.1: ref %array_type = value_as_ref %T.ref.loc12_60
// CHECK:STDOUT:       %.loc12_63.2: ref type = array_index %.loc12_63.1, %.loc12_62.2
// CHECK:STDOUT:       %.loc12_63.3: type = acquire_value %.loc12_63.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %array_type) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.dcb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.4c7)]
// CHECK:STDOUT:   %T.loc4_18.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.9b7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.881
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_7.2: %array_type) {
// CHECK:STDOUT:   %T.patt.loc12_7.2: %pattern_type.dcb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.4c7)]
// CHECK:STDOUT:   %T.loc12_7.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:   %HoldsType.loc12_47.1: type = class_type @HoldsType, @HoldsType(%T.loc12_7.1) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.881)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc12_47.1 [symbolic = %pattern_type (constants.%pattern_type.d35)]
// CHECK:STDOUT:   %x.param_patt.loc12_34.2: @F.%pattern_type (%pattern_type.d35) = value_param_pattern [symbolic = %x.param_patt.loc12_34.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc12_34.2: @F.%pattern_type (%pattern_type.d35) = at_binding_pattern x, %x.param_patt.loc12_34.2 [symbolic = %x.patt.loc12_34.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc12_47.1 [symbolic = %require_complete (constants.%require_complete.80d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc12_47.1 (%HoldsType.881), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc17_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.1: %tuple.type.85c = tuple_literal (%C.ref) [concrete = constants.%tuple.e45]
// CHECK:STDOUT:   %.loc17_36: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %.loc17_36 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0: %.524 = impl_witness_access constants.%Copy.impl_witness.638, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc17_25.3: ref type = array_index %.loc17_25.2, %int_0
// CHECK:STDOUT:   %.loc17_25.4: init type to %.loc17_25.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.5: init %array_type to %.loc17_25.2 = array_init (%.loc17_25.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.1: init %array_type = converted %.loc17_25.1, %.loc17_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.2: ref %array_type = temporary %.loc17_25.2, %.loc17_27.1 [concrete = constants.%.1c6]
// CHECK:STDOUT:   %.loc17_27.3: %array_type = acquire_value %.loc17_27.2 [concrete = constants.%array]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%array) [concrete = constants.%HoldsType.f11]
// CHECK:STDOUT:   %.loc17_6.2: ref %HoldsType.f11 = temporary_storage
// CHECK:STDOUT:   %.loc17_6.3: init %HoldsType.f11 to %.loc17_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_8: init %HoldsType.f11 = converted %.loc17_6.1, %.loc17_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_48: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.1c6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_27.1(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_27.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.9b7) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.4c7
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.9b7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.9b7) {
// CHECK:STDOUT:   %T.patt.loc12_7.2 => constants.%T.patt.4c7
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T.9b7
// CHECK:STDOUT:   %HoldsType.loc12_47.1 => constants.%HoldsType.881
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d35
// CHECK:STDOUT:   %x.param_patt.loc12_34.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc12_34.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%array) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.4c7
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%array
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
