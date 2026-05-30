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
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.b25: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1d5: type = pattern_type %HoldsType.b25 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.071: <witness> = require_complete_type %HoldsType.b25 [symbolic]
// CHECK:STDOUT:   %require_complete.e7a: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.1e5: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.f41: type = class_type @HoldsType, @HoldsType(%tuple.1e5) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.f41 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.19e: type = pattern_type %HoldsType.f41 [concrete]
// CHECK:STDOUT:   %pattern_type.90b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.1e5) [concrete]
// CHECK:STDOUT:   %.fce: ref %HoldsType.f41 = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.521: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc13_30.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.65d: <bound method> = bound_method %.521, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.70b: <bound method> = bound_method %.fce, %Destroy.Op.1dc86d.3 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_28 (%pattern_type.1d5) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_28 (%pattern_type.1d5) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_52 (%pattern_type.e66) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_52 (%pattern_type.e66) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_11: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc8_17.2: %tuple.type = tuple_literal (%.loc8_11) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc8_17.3: type = converted %.loc8_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.b25) = value_param call_param0
// CHECK:STDOUT:     %.loc8_41: type = splice_block %HoldsType.loc8_41.2 [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.b25)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_40: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_41.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.b25)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_41.1 (%HoldsType.b25) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_55: type = splice_block %tuple.elem0.loc8_55.2 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_54: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc8_55.2: type = tuple_access %T.ref.loc8_54, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc4_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.b25
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
// CHECK:STDOUT:   %T.loc8_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_41.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.b25)]
// CHECK:STDOUT:   %pattern_type.loc8_28: type = pattern_type %HoldsType.loc8_41.1 [symbolic = %pattern_type.loc8_28 (constants.%pattern_type.1d5)]
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1: type = tuple_access %T.loc8_7.1, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_52: type = pattern_type %tuple.elem0.loc8_55.1 [symbolic = %pattern_type.loc8_52 (constants.%pattern_type.e66)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_28: <witness> = require_complete_type %HoldsType.loc8_41.1 [symbolic = %require_complete.loc8_28 (constants.%require_complete.071)]
// CHECK:STDOUT:   %require_complete.loc8_52: <witness> = require_complete_type %tuple.elem0.loc8_55.1 [symbolic = %require_complete.loc8_52 (constants.%require_complete.e7a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.b25), %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0)) {
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
// CHECK:STDOUT:   %.loc13_25: %tuple.type = tuple_literal (%C.ref) [concrete = constants.%tuple.1e5]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple.1e5]
// CHECK:STDOUT:   %.loc13_26: %tuple.type = converted %.loc13_25, %tuple [concrete = constants.%tuple.1e5]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.1e5) [concrete = constants.%HoldsType.f41]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.f41 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.f41 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.f41 = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.1e5) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.f41 = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.fce]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.f41 = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C to %.loc13_30.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_30.1, %.loc13_30.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.5: ref %C = temporary %.loc13_30.2, %.loc13_30.4 [concrete = constants.%.521]
// CHECK:STDOUT:   %.loc13_30.6: %C = acquire_value %.loc13_30.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_30.6)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_30: init %empty_tuple.type = call constants.%Destroy.Op.bound.65d(constants.%.521)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.Op.bound.70b(constants.%.fce)
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
// CHECK:STDOUT: fn @Destroy.Op.loc13_8(%self.param: ref %HoldsType.f41) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.b25
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.1d5
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_52 => constants.%pattern_type.e66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.1e5) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%tuple.1e5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.1e5) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%tuple.1e5
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.f41
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.19e
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_52 => constants.%pattern_type.90b
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
// CHECK:STDOUT:   %T: %struct_type.t = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.ba6: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.af1: type = pattern_type %HoldsType.ba6 [symbolic]
// CHECK:STDOUT:   %.424: type = struct_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.92c: type = pattern_type %.424 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.225: <witness> = require_complete_type %HoldsType.ba6 [symbolic]
// CHECK:STDOUT:   %require_complete.ba0: <witness> = require_complete_type %.424 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.39e: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.39e = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.551: type = pattern_type %HoldsType.39e [concrete]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %.971: ref %HoldsType.39e = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.62f: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc13_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.dd8: <bound method> = bound_method %.62f, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.b40: <bound method> = bound_method %.971, %Destroy.Op.1dc86d.3 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_30: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_26: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_30 (%pattern_type.af1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_30 (%pattern_type.af1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_54 (%pattern_type.92c) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_54 (%pattern_type.92c) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_15: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.ba6) = value_param call_param0
// CHECK:STDOUT:     %.loc8_43: type = splice_block %HoldsType.loc8_43.2 [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.ba6)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_43.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.ba6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_43.1 (%HoldsType.ba6) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_57.1 (%.424) = value_param call_param1
// CHECK:STDOUT:     %.loc8_57.2: type = splice_block %.loc8_57.3 [symbolic = %.loc8_57.1 (constants.%.424)] {
// CHECK:STDOUT:       %T.ref.loc8_56: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_57.3: type = struct_access %T.ref.loc8_56, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_57.1 (%.424) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %struct_type.t) {
// CHECK:STDOUT:   %T.loc4_18.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.ba6
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
// CHECK:STDOUT:   %T.loc8_7.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_43.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.ba6)]
// CHECK:STDOUT:   %pattern_type.loc8_30: type = pattern_type %HoldsType.loc8_43.1 [symbolic = %pattern_type.loc8_30 (constants.%pattern_type.af1)]
// CHECK:STDOUT:   %.loc8_57.1: type = struct_access %T.loc8_7.1, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:   %pattern_type.loc8_54: type = pattern_type %.loc8_57.1 [symbolic = %pattern_type.loc8_54 (constants.%pattern_type.92c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_30: <witness> = require_complete_type %HoldsType.loc8_43.1 [symbolic = %require_complete.loc8_30 (constants.%require_complete.225)]
// CHECK:STDOUT:   %require_complete.loc8_54: <witness> = require_complete_type %.loc8_57.1 [symbolic = %require_complete.loc8_54 (constants.%require_complete.ba0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.ba6), %a.param: @F.%.loc8_57.1 (%.424)) {
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
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.39e]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.39e = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.39e to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.39e = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.39e = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.971]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.39e = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C to %.loc13_33.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: init %C = converted %.loc13_33.1, %.loc13_33.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.5: ref %C = temporary %.loc13_33.2, %.loc13_33.4 [concrete = constants.%.62f]
// CHECK:STDOUT:   %.loc13_33.6: %C = acquire_value %.loc13_33.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_33.6)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_33: init %empty_tuple.type = call constants.%Destroy.Op.bound.dd8(constants.%.62f)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.Op.bound.b40(constants.%.971)
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
// CHECK:STDOUT: fn @Destroy.Op.loc13_8(%self.param: ref %HoldsType.39e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.ba6
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.af1
// CHECK:STDOUT:   %.loc8_57.1 => constants.%.424
// CHECK:STDOUT:   %pattern_type.loc8_54 => constants.%pattern_type.92c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%struct) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%struct
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.39e
// CHECK:STDOUT:   %pattern_type.loc8_30 => constants.%pattern_type.551
// CHECK:STDOUT:   %.loc8_57.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_54 => constants.%pattern_type.65c
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
// CHECK:STDOUT:   %pattern_type.a14: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %T.ea6: %Class = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.794ed1.1: type = class_type @HoldsType, @HoldsType(%T.ea6) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.0890cb.1: type = pattern_type %HoldsType.794ed1.1 [symbolic]
// CHECK:STDOUT:   %.b5360e.1: ref type = class_element_access %T.ea6, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.3ea1ee.1: <witness> = require_complete_type %HoldsType.794ed1.1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %c: %Class = symbolic_binding c, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.794ed1.2: type = class_type @HoldsType, @HoldsType(%c) [symbolic]
// CHECK:STDOUT:   %require_complete.3ea1ee.2: <witness> = require_complete_type %HoldsType.794ed1.2 [symbolic]
// CHECK:STDOUT:   %HoldsType.val.b6a: %HoldsType.794ed1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.0890cb.2: type = pattern_type %HoldsType.794ed1.2 [symbolic]
// CHECK:STDOUT:   %.b5360e.2: ref type = class_element_access %c, element0 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.36f: <specific function> = specific_function %F, @F(%c) [symbolic]
// CHECK:STDOUT:   %.38d: ref %HoldsType.794ed1.2 = temporary invalid, %HoldsType.val.b6a [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %HoldsType.794ed1.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.f51: %Destroy.type = facet_value %HoldsType.794ed1.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9ef: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f51) [symbolic]
// CHECK:STDOUT:   %.08e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9ef, %Destroy.facet.f51 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a4f: %.08e = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.54f: <bound method> = bound_method %.38d, %impl.elem0.a4f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9c0: <specific function> = specific_impl_function %impl.elem0.a4f, @Destroy.WithSelf.Op(%Destroy.facet.f51) [symbolic]
// CHECK:STDOUT:   %bound_method.b2d: <bound method> = bound_method %.38d, %specific_impl_fn.9c0 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.592: <witness> = impl_witness imports.%Copy.impl_witness_table.fbd [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.592) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b33: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.173: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b33, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %.db0: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%Class.val) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc30_13.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.411: <bound method> = bound_method %.db0, %Destroy.Op.1dc86d.4 [concrete]
// CHECK:STDOUT:   %HoldsType.36e: type = class_type @HoldsType, @HoldsType(%Class.val) [concrete]
// CHECK:STDOUT:   %HoldsType.val.4d2: %HoldsType.36e = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.5c7: type = pattern_type %HoldsType.36e [concrete]
// CHECK:STDOUT:   %F.specific_fn.a2e: <specific function> = specific_function %F, @F(%Class.val) [concrete]
// CHECK:STDOUT:   %.dc0: ref %HoldsType.36e = temporary invalid, %HoldsType.val.4d2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc26_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.340963.5: <witness> = custom_witness (%Destroy.Op.1dc86d.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5f2: %Destroy.type = facet_value %HoldsType.36e, (%custom_witness.340963.5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.751: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5f2) [concrete]
// CHECK:STDOUT:   %.789: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.751, %Destroy.facet.5f2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.6c3: <bound method> = bound_method %.dc0, %Destroy.Op.1dc86d.5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.41e: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.fbd = impl_witness_table (%Core.import_ref.41e), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.a14 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_18.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_18.1 (constants.%T.ea6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a14 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.0890cb.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.0890cb.1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: <error> = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_10: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_7.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.ea6)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.794ed1.1) = value_param call_param0
// CHECK:STDOUT:     %.loc21_38: type = splice_block %HoldsType.loc21_38.2 [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.794ed1.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc21_37: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.ea6)]
// CHECK:STDOUT:       %HoldsType.loc21_38.2: type = class_type @HoldsType, @HoldsType(constants.%T.ea6) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.794ed1.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc21_38.1 (%HoldsType.794ed1.1) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc21_51: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.ea6)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:       %.loc21_52.2: ref type = class_element_access %T.ref.loc21_51, element0 [symbolic = %.loc21_52.1 (constants.%.b5360e.1)]
// CHECK:STDOUT:       %.loc21_52.3: type = acquire_value %.loc21_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %pattern_type.a14 = symbolic_binding_pattern c, 0 [concrete]
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
// CHECK:STDOUT:   %T.loc8_18.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_18.1 (constants.%T.ea6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.794ed1.1
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
// CHECK:STDOUT:   %T.loc21_7.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.ea6)]
// CHECK:STDOUT:   %HoldsType.loc21_38.1: type = class_type @HoldsType, @HoldsType(%T.loc21_7.1) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.794ed1.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc21_38.1 [symbolic = %pattern_type (constants.%pattern_type.0890cb.1)]
// CHECK:STDOUT:   %.loc21_52.1: ref type = class_element_access %T.loc21_7.1, element0 [symbolic = %.loc21_52.1 (constants.%.b5360e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc21_38.1 [symbolic = %require_complete (constants.%require_complete.3ea1ee.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.794ed1.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%c.loc25_7.2: %Class) {
// CHECK:STDOUT:   %c.loc25_7.1: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_7.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2: type = class_type @HoldsType, @HoldsType(%c.loc25_7.1) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.794ed1.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc26_22.2 [symbolic = %require_complete (constants.%require_complete.3ea1ee.2)]
// CHECK:STDOUT:   %HoldsType.val: @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = struct_value () [symbolic = %HoldsType.val (constants.%HoldsType.val.b6a)]
// CHECK:STDOUT:   %F.specific_fn.loc26_3.2: <specific function> = specific_function constants.%F, @F(%c.loc25_7.1) [symbolic = %F.specific_fn.loc26_3.2 (constants.%F.specific_fn.36f)]
// CHECK:STDOUT:   %.loc26_8.4: ref @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = temporary invalid, %HoldsType.val [symbolic = %.loc26_8.4 (constants.%.38d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %HoldsType.loc26_22.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %HoldsType.loc26_22.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.f51)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9ef)]
// CHECK:STDOUT:   %.loc26_8.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc26_8.5 (constants.%.08e)]
// CHECK:STDOUT:   %impl.elem0.loc26_8.2: @G.%.loc26_8.5 (%.08e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_8.2 (constants.%impl.elem0.a4f)]
// CHECK:STDOUT:   %bound_method.loc26_8.3: <bound method> = bound_method %.loc26_8.4, %impl.elem0.loc26_8.2 [symbolic = %bound_method.loc26_8.3 (constants.%bound_method.54f)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_8.2: <specific function> = specific_impl_function %impl.elem0.loc26_8.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc26_8.2 (constants.%specific_impl_fn.9c0)]
// CHECK:STDOUT:   %bound_method.loc26_8.4: <bound method> = bound_method %.loc26_8.4, %specific_impl_fn.loc26_8.2 [symbolic = %bound_method.loc26_8.4 (constants.%bound_method.b2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc26_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:     %c.ref: %Class = name_ref c, %c.loc25_7.2 [symbolic = %c.loc25_7.1 (constants.%c)]
// CHECK:STDOUT:     %HoldsType.loc26_22.1: type = class_type @HoldsType, @HoldsType(constants.%c) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.794ed1.2)]
// CHECK:STDOUT:     %.loc26_6.2: ref @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = temporary_storage
// CHECK:STDOUT:     %.loc26_6.3: init @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) to %.loc26_6.2 = class_init () [symbolic = %HoldsType.val (constants.%HoldsType.val.b6a)]
// CHECK:STDOUT:     %.loc26_8.1: init @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = converted %.loc26_6.1, %.loc26_6.3 [symbolic = %HoldsType.val (constants.%HoldsType.val.b6a)]
// CHECK:STDOUT:     %.loc26_26: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %F.specific_fn.loc26_3.1: <specific function> = specific_function %F.ref, @F(constants.%c) [symbolic = %F.specific_fn.loc26_3.2 (constants.%F.specific_fn.36f)]
// CHECK:STDOUT:     %.loc26_8.2: ref @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = temporary %.loc26_6.2, %.loc26_8.1 [symbolic = %.loc26_8.4 (constants.%.38d)]
// CHECK:STDOUT:     %.loc26_8.3: @G.%HoldsType.loc26_22.2 (%HoldsType.794ed1.2) = acquire_value %.loc26_8.2 [symbolic = %HoldsType.val (constants.%HoldsType.val.b6a)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc26_3.1(%.loc26_8.3, <error>)
// CHECK:STDOUT:     %impl.elem0.loc26_8.1: @G.%.loc26_8.5 (%.08e) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_8.2 (constants.%impl.elem0.a4f)]
// CHECK:STDOUT:     %bound_method.loc26_8.1: <bound method> = bound_method %.loc26_8.2, %impl.elem0.loc26_8.1 [symbolic = %bound_method.loc26_8.3 (constants.%bound_method.54f)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_8.1: <specific function> = specific_impl_function %impl.elem0.loc26_8.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.f51) [symbolic = %specific_impl_fn.loc26_8.2 (constants.%specific_impl_fn.9c0)]
// CHECK:STDOUT:     %bound_method.loc26_8.2: <bound method> = bound_method %.loc26_8.2, %specific_impl_fn.loc26_8.1 [symbolic = %bound_method.loc26_8.4 (constants.%bound_method.b2d)]
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc26_8.2(%.loc26_8.2)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.1: %struct_type.t = struct_literal (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.173 = impl_witness_access constants.%Copy.impl_witness.592, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc30_12.3: ref type = class_element_access %.loc30_12.2, element0
// CHECK:STDOUT:   %.loc30_12.4: init type to %.loc30_12.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_12.5: init %Class to %.loc30_12.2 = class_init (%.loc30_12.4) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc30_13.1: init %Class = converted %.loc30_12.1, %.loc30_12.5 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc30_13.2: ref %Class = temporary %.loc30_12.2, %.loc30_13.1 [concrete = constants.%.db0]
// CHECK:STDOUT:   %.loc30_13.3: %Class = acquire_value %.loc30_13.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%Class.val) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound.411(constants.%.db0)
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
// CHECK:STDOUT: fn @Destroy.Op.loc26_8.2(%self.param: ref %HoldsType.36e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.ea6) {
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%T.ea6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.ea6) {
// CHECK:STDOUT:   %T.loc21_7.1 => constants.%T.ea6
// CHECK:STDOUT:   %HoldsType.loc21_38.1 => constants.%HoldsType.794ed1.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0890cb.1
// CHECK:STDOUT:   %.loc21_52.1 => constants.%.b5360e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%c) {
// CHECK:STDOUT:   %c.loc25_7.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%c) {
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %T.loc21_7.1 => constants.%c
// CHECK:STDOUT:   %HoldsType.loc21_38.1 => constants.%HoldsType.794ed1.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0890cb.2
// CHECK:STDOUT:   %.loc21_52.1 => constants.%.b5360e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.3ea1ee.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Class.val) {
// CHECK:STDOUT:   %c.loc25_7.1 => constants.%Class.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2 => constants.%HoldsType.36e
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %HoldsType.val => constants.%HoldsType.val.4d2
// CHECK:STDOUT:   %F.specific_fn.loc26_3.2 => constants.%F.specific_fn.a2e
// CHECK:STDOUT:   %.loc26_8.4 => constants.%.dc0
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.340963.5
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.5f2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.751
// CHECK:STDOUT:   %.loc26_8.5 => constants.%.789
// CHECK:STDOUT:   %impl.elem0.loc26_8.2 => constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %bound_method.loc26_8.3 => constants.%Destroy.Op.bound.6c3
// CHECK:STDOUT:   %specific_impl_fn.loc26_8.2 => constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %bound_method.loc26_8.4 => constants.%Destroy.Op.bound.6c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%Class.val) {
// CHECK:STDOUT:   %T.loc8_18.1 => constants.%Class.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Class.val) {
// CHECK:STDOUT:   %T.loc21_7.1 => constants.%Class.val
// CHECK:STDOUT:   %HoldsType.loc21_38.1 => constants.%HoldsType.36e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5c7
// CHECK:STDOUT:   %.loc21_52.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
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
// CHECK:STDOUT:   %T.9b7: %array_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.6d0: type = class_type @HoldsType, @HoldsType(%T.9b7) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fbd: type = pattern_type %HoldsType.6d0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.07b: <witness> = require_complete_type %HoldsType.6d0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.3c6: %tuple.type.85c = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.592: <witness> = impl_witness imports.%Copy.impl_witness_table.fbd [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.592) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b33: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.173: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b33, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %.3fa: ref %array_type = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %HoldsType.c6c: type = class_type @HoldsType, @HoldsType(%array) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.c6c = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.246: type = pattern_type %HoldsType.c6c [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%array) [concrete]
// CHECK:STDOUT:   %.01d: ref %HoldsType.c6c = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc17_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.862: <bound method> = bound_method %.01d, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc17_27.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.5f8: <bound method> = bound_method %.3fa, %Destroy.Op.1dc86d.4 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.41e: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.fbd = impl_witness_table (%Core.import_ref.41e), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.dcb = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.dcb = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.fbd) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.fbd) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: <error> = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_16: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc12_16 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc12_47.1 (%HoldsType.6d0) = value_param call_param0
// CHECK:STDOUT:     %.loc12_47: type = splice_block %HoldsType.loc12_47.2 [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.6d0)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc12_46: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %HoldsType.loc12_47.2: type = class_type @HoldsType, @HoldsType(constants.%T.9b7) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.6d0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc12_47.1 (%HoldsType.6d0) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc12_60: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:       %bound_method.loc12_62.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_62.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_62.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %.loc12_62.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %.loc12_62.2: %i32 = converted %int_0, %.loc12_62.1 [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %.loc12_63.1: ref %array_type = value_as_ref %T.ref.loc12_60
// CHECK:STDOUT:       %.loc12_63.2: ref type = array_index %.loc12_63.1, %.loc12_62.2
// CHECK:STDOUT:       %.loc12_63.3: type = acquire_value %.loc12_63.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %array_type) {
// CHECK:STDOUT:   %T.loc4_18.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.9b7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.6d0
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
// CHECK:STDOUT:   %T.loc12_7.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.9b7)]
// CHECK:STDOUT:   %HoldsType.loc12_47.1: type = class_type @HoldsType, @HoldsType(%T.loc12_7.1) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType.6d0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc12_47.1 [symbolic = %pattern_type (constants.%pattern_type.fbd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc12_47.1 [symbolic = %require_complete (constants.%require_complete.07b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc12_47.1 (%HoldsType.6d0), %a.param: <error>) {
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
// CHECK:STDOUT:   %.loc17_25.1: %tuple.type.85c = tuple_literal (%C.ref) [concrete = constants.%tuple.3c6]
// CHECK:STDOUT:   %.loc17_36: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %.loc17_36 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0: %.173 = impl_witness_access constants.%Copy.impl_witness.592, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc17_25.3: ref type = array_index %.loc17_25.2, %int_0
// CHECK:STDOUT:   %.loc17_25.4: init type to %.loc17_25.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.5: init %array_type to %.loc17_25.2 = array_init (%.loc17_25.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.1: init %array_type = converted %.loc17_25.1, %.loc17_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.2: ref %array_type = temporary %.loc17_25.2, %.loc17_27.1 [concrete = constants.%.3fa]
// CHECK:STDOUT:   %.loc17_27.3: %array_type = acquire_value %.loc17_27.2 [concrete = constants.%array]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%array) [concrete = constants.%HoldsType.c6c]
// CHECK:STDOUT:   %.loc17_6.2: ref %HoldsType.c6c = temporary_storage
// CHECK:STDOUT:   %.loc17_6.3: init %HoldsType.c6c to %.loc17_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_8.1: init %HoldsType.c6c = converted %.loc17_6.1, %.loc17_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_48: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%array) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc17_8.2: ref %HoldsType.c6c = temporary %.loc17_6.2, %.loc17_8.1 [concrete = constants.%.01d]
// CHECK:STDOUT:   %.loc17_8.3: %HoldsType.c6c = acquire_value %.loc17_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc17_8.3, <error>)
// CHECK:STDOUT:   %Destroy.Op.call.loc17_8: init %empty_tuple.type = call constants.%Destroy.Op.bound.862(constants.%.01d)
// CHECK:STDOUT:   %Destroy.Op.call.loc17_27: init %empty_tuple.type = call constants.%Destroy.Op.bound.5f8(constants.%.3fa)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_8.2(%self.param: ref %HoldsType.c6c) {
// CHECK:STDOUT: !entry:
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
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.9b7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.9b7) {
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T.9b7
// CHECK:STDOUT:   %HoldsType.loc12_47.1 => constants.%HoldsType.6d0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fbd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%array) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%array
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%array) {
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%array
// CHECK:STDOUT:   %HoldsType.loc12_47.1 => constants.%HoldsType.c6c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.246
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
