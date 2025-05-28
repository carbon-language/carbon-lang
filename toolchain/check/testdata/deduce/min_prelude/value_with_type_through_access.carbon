// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/min_prelude/value_with_type_through_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/min_prelude/value_with_type_through_access.carbon

// --- tuple_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! (type, )) {}

// `a` is received as a value, and has a type that requires going through
// TupleAccess. Building the value representation must handle this indirection.
//@dump-sem-ir-begin
fn F[T:! (type, )](x: HoldsType(T), a: T.0) {}
//@dump-sem-ir-end

class C {}

fn G() {
  //@dump-sem-ir-begin
  F({} as HoldsType((C, )), {});
  //@dump-sem-ir-end
}

// --- struct_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! {.t: type}) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
//@dump-sem-ir-begin
fn F[T:! {.t: type}](x: HoldsType(T), a: T.t) {}
//@dump-sem-ir-end

class C {}

fn G() {
  //@dump-sem-ir-begin
  F({} as HoldsType({.t = C}), {});
  //@dump-sem-ir-end
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
// ClassElementAccess but it is a BindValue, which has a runtime value.
//
// CHECK:STDERR: fail_todo_class_access.carbon:[[@LINE+4]]:37: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T:! Class](x: HoldsType(T), a: T.t) {}
// CHECK:STDERR:                                     ^~~
// CHECK:STDERR:
fn F[T:! Class](x: HoldsType(T), a: T.t) {}

class C {}

fn G() {
  let c:! Class = {.t = C} as Class;
  F({} as HoldsType(c), {});
}

// --- fail_todo_array_index.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! array(type, 1)) {}

// `a` is received as a value, and has a type that requires going through
// ArrayIndex. Building the value representation must handle this indirection.
// CHECK:STDERR: fail_todo_array_index.carbon:[[@LINE+4]]:46: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T:! array(type, 1)](x: HoldsType(T), a: T[0]) {}
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
fn F[T:! array(type, 1)](x: HoldsType(T), a: T[0]) {}

class C {}

fn G() {
  // CHECK:STDERR: fail_todo_array_index.carbon:[[@LINE+7]]:11: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   F({} as HoldsType((C, ) as array(type, 1)), {});
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_array_index.carbon:[[@LINE-16]]:17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class HoldsType(T:! array(type, 1)) {}
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  F({} as HoldsType((C, ) as array(type, 1)), {});
}


// CHECK:STDOUT: --- tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %T: %tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.cc9: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.ec6: type = pattern_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.08e: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.514: <witness> = require_complete_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %require_complete.fec: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.c09: type = class_type @HoldsType, @HoldsType(%tuple) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.c09 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8e0: type = pattern_type %HoldsType.c09 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc9_20 (%pattern_type.ec6) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.ec6) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc9_37 (%pattern_type.08e) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc9_37 (%pattern_type.08e) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.loc9_17.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc9_17.3: type = converted %.loc9_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc9_34.2 (%HoldsType.cc9) = value_param call_param0
// CHECK:STDOUT:     %.loc9_34: type = splice_block %HoldsType.loc9_34.1 [symbolic = %HoldsType.loc9_34.2 (constants.%HoldsType.cc9)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc9_33: %tuple.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc9_34.1: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc9_34.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc9_34.2 (%HoldsType.cc9) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc9_41.2 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc9_41: type = splice_block %tuple.elem0.loc9_41.1 [symbolic = %tuple.elem0.loc9_41.2 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc9_40: %tuple.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc9_41.1: type = tuple_access %T.ref.loc9_40, element0 [symbolic = %tuple.elem0.loc9_41.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc9_41.2 (%tuple.elem0) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %tuple.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc9_34.2: type = class_type @HoldsType, @HoldsType(%T.loc9_6.2) [symbolic = %HoldsType.loc9_34.2 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %HoldsType.loc9_34.2 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.ec6)]
// CHECK:STDOUT:   %tuple.elem0.loc9_41.2: type = tuple_access %T.loc9_6.2, element0 [symbolic = %tuple.elem0.loc9_41.2 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc9_37: type = pattern_type %tuple.elem0.loc9_41.2 [symbolic = %pattern_type.loc9_37 (constants.%pattern_type.08e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_21: <witness> = require_complete_type %HoldsType.loc9_34.2 [symbolic = %require_complete.loc9_21 (constants.%require_complete.514)]
// CHECK:STDOUT:   %require_complete.loc9_38: <witness> = require_complete_type %tuple.elem0.loc9_41.2 [symbolic = %require_complete.loc9_38 (constants.%require_complete.fec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc9_34.2 (%HoldsType.cc9), %a.param: @F.%tuple.elem0.loc9_41.2 (%tuple.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc16_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc16_25: %tuple.type = tuple_literal (%C.ref)
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc16_26: %tuple.type = converted %.loc16_25, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple) [concrete = constants.%HoldsType.c09]
// CHECK:STDOUT:   %.loc16_6.2: ref %HoldsType.c09 = temporary_storage
// CHECK:STDOUT:   %.loc16_6.3: init %HoldsType.c09 = class_init (), %.loc16_6.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc16_6.4: ref %HoldsType.c09 = temporary %.loc16_6.2, %.loc16_6.3
// CHECK:STDOUT:   %.loc16_8.1: ref %HoldsType.c09 = converted %.loc16_6.1, %.loc16_6.4
// CHECK:STDOUT:   %.loc16_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc16_8.2: %HoldsType.c09 = bind_value %.loc16_8.1
// CHECK:STDOUT:   %.loc16_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_30.3: init %C = class_init (), %.loc16_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_30.4: ref %C = temporary %.loc16_30.2, %.loc16_30.3
// CHECK:STDOUT:   %.loc16_30.5: ref %C = converted %.loc16_30.1, %.loc16_30.4
// CHECK:STDOUT:   %.loc16_30.6: %C = bind_value %.loc16_30.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc16_8.2, %.loc16_30.6)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc9_34.2 => constants.%HoldsType.cc9
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.ec6
// CHECK:STDOUT:   %tuple.elem0.loc9_41.2 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc9_37 => constants.%pattern_type.08e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%tuple
// CHECK:STDOUT:   %HoldsType.loc9_34.2 => constants.%HoldsType.c09
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.8e0
// CHECK:STDOUT:   %tuple.elem0.loc9_41.2 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc9_37 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_21 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc9_38 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %T: %struct_type.t = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f2: type = pattern_type %struct_type.t [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.843: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.aa4: type = pattern_type %HoldsType.843 [symbolic]
// CHECK:STDOUT:   %.20a: type = struct_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9f0: type = pattern_type %.20a [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b19: <witness> = require_complete_type %HoldsType.843 [symbolic]
// CHECK:STDOUT:   %require_complete.1b9: <witness> = require_complete_type %.20a [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.705: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.705 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.50b: type = pattern_type %HoldsType.705 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc9_22 (%pattern_type.aa4) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc9_22 (%pattern_type.aa4) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc9_39 (%pattern_type.9f0) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc9_39 (%pattern_type.9f0) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     %T.loc9_6.1: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc9_36.2 (%HoldsType.843) = value_param call_param0
// CHECK:STDOUT:     %.loc9_36: type = splice_block %HoldsType.loc9_36.1 [symbolic = %HoldsType.loc9_36.2 (constants.%HoldsType.843)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc9_35: %struct_type.t = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc9_36.1: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc9_36.2 (constants.%HoldsType.843)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc9_36.2 (%HoldsType.843) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc9_43.3 (%.20a) = value_param call_param1
// CHECK:STDOUT:     %.loc9_43.1: type = splice_block %.loc9_43.2 [symbolic = %.loc9_43.3 (constants.%.20a)] {
// CHECK:STDOUT:       %T.ref.loc9_42: %struct_type.t = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %.loc9_43.2: type = struct_access %T.ref.loc9_42, element0 [symbolic = %.loc9_43.3 (constants.%.20a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc9_43.3 (%.20a) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %struct_type.t) {
// CHECK:STDOUT:   %T.loc9_6.2: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc9_36.2: type = class_type @HoldsType, @HoldsType(%T.loc9_6.2) [symbolic = %HoldsType.loc9_36.2 (constants.%HoldsType.843)]
// CHECK:STDOUT:   %pattern_type.loc9_22: type = pattern_type %HoldsType.loc9_36.2 [symbolic = %pattern_type.loc9_22 (constants.%pattern_type.aa4)]
// CHECK:STDOUT:   %.loc9_43.3: type = struct_access %T.loc9_6.2, element0 [symbolic = %.loc9_43.3 (constants.%.20a)]
// CHECK:STDOUT:   %pattern_type.loc9_39: type = pattern_type %.loc9_43.3 [symbolic = %pattern_type.loc9_39 (constants.%pattern_type.9f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_23: <witness> = require_complete_type %HoldsType.loc9_36.2 [symbolic = %require_complete.loc9_23 (constants.%require_complete.b19)]
// CHECK:STDOUT:   %require_complete.loc9_40: <witness> = require_complete_type %.loc9_43.3 [symbolic = %require_complete.loc9_40 (constants.%require_complete.1b9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc9_36.2 (%HoldsType.843), %a.param: @F.%.loc9_43.3 (%.20a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc16_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc16_28: %struct_type.t = struct_literal (%C.ref)
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_29: %struct_type.t = converted %.loc16_28, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.705]
// CHECK:STDOUT:   %.loc16_6.2: ref %HoldsType.705 = temporary_storage
// CHECK:STDOUT:   %.loc16_6.3: init %HoldsType.705 = class_init (), %.loc16_6.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc16_6.4: ref %HoldsType.705 = temporary %.loc16_6.2, %.loc16_6.3
// CHECK:STDOUT:   %.loc16_8.1: ref %HoldsType.705 = converted %.loc16_6.1, %.loc16_6.4
// CHECK:STDOUT:   %.loc16_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc16_8.2: %HoldsType.705 = bind_value %.loc16_8.1
// CHECK:STDOUT:   %.loc16_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_33.3: init %C = class_init (), %.loc16_33.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc16_33.4: ref %C = temporary %.loc16_33.2, %.loc16_33.3
// CHECK:STDOUT:   %.loc16_33.5: ref %C = converted %.loc16_33.1, %.loc16_33.4
// CHECK:STDOUT:   %.loc16_33.6: %C = bind_value %.loc16_33.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc16_8.2, %.loc16_33.6)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc9_36.2 => constants.%HoldsType.843
// CHECK:STDOUT:   %pattern_type.loc9_22 => constants.%pattern_type.aa4
// CHECK:STDOUT:   %.loc9_43.3 => constants.%.20a
// CHECK:STDOUT:   %pattern_type.loc9_39 => constants.%pattern_type.9f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%struct
// CHECK:STDOUT:   %HoldsType.loc9_36.2 => constants.%HoldsType.705
// CHECK:STDOUT:   %pattern_type.loc9_22 => constants.%pattern_type.50b
// CHECK:STDOUT:   %.loc9_43.3 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc9_39 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_23 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc9_40 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
