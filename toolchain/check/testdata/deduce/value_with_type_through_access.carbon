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
fn F[T:! (type, )](x: HoldsType(T), a: T.0) {}

class C {}

fn G() {
  F({} as HoldsType((C, )), {});
}

// --- struct_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T:! {.t: type}) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
fn F[T:! {.t: type}](x: HoldsType(T), a: T.t) {}

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
// CHECK:STDOUT:   %T.6eb: %tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.cc9: type = class_type @HoldsType, @HoldsType(%T.6eb) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.ec6: type = pattern_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T.6eb, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.08e: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.514: <witness> = require_complete_type %HoldsType.cc9 [symbolic]
// CHECK:STDOUT:   %require_complete.fec: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.c09: type = class_type @HoldsType, @HoldsType(%tuple) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.c09 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8e0: type = pattern_type %HoldsType.c09 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.1, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6eb: <specific function> = specific_function %Op.5d7, @Op.1(%C) [concrete]
// CHECK:STDOUT:   %Op.type.431: type = fn_type @Op.1, @Destroy.impl(%HoldsType.c09) [concrete]
// CHECK:STDOUT:   %Op.f20: %Op.type.431 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79a: type = ptr_type %HoldsType.c09 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.58d: <specific function> = specific_function %Op.f20, @Op.1(%HoldsType.c09) [concrete]
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
// CHECK:STDOUT:       %.loc4_28.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc4_28.3: type = converted %.loc4_28.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.6eb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_20 (%pattern_type.ec6) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_20 (%pattern_type.ec6) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_37 (%pattern_type.08e) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_37 (%pattern_type.08e) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.loc8_17.2: %tuple.type = tuple_literal (type)
// CHECK:STDOUT:       %.loc8_17.3: type = converted %.loc8_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.6eb)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_34.1 (%HoldsType.cc9) = value_param call_param0
// CHECK:STDOUT:     %.loc8_34: type = splice_block %HoldsType.loc8_34.2 [symbolic = %HoldsType.loc8_34.1 (constants.%HoldsType.cc9)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_33: %tuple.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.6eb)]
// CHECK:STDOUT:       %HoldsType.loc8_34.2: type = class_type @HoldsType, @HoldsType(constants.%T.6eb) [symbolic = %HoldsType.loc8_34.1 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_34.1 (%HoldsType.cc9) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_41.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_41: type = splice_block %tuple.elem0.loc8_41.2 [symbolic = %tuple.elem0.loc8_41.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_40: %tuple.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.6eb)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc8_41.2: type = tuple_access %T.ref.loc8_40, element0 [symbolic = %tuple.elem0.loc8_41.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc8_41.1 (%tuple.elem0) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc4_17.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.6eb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.cc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.6eb)]
// CHECK:STDOUT:   %HoldsType.loc8_34.1: type = class_type @HoldsType, @HoldsType(%T.loc8_6.1) [symbolic = %HoldsType.loc8_34.1 (constants.%HoldsType.cc9)]
// CHECK:STDOUT:   %pattern_type.loc8_20: type = pattern_type %HoldsType.loc8_34.1 [symbolic = %pattern_type.loc8_20 (constants.%pattern_type.ec6)]
// CHECK:STDOUT:   %tuple.elem0.loc8_41.1: type = tuple_access %T.loc8_6.1, element0 [symbolic = %tuple.elem0.loc8_41.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_37: type = pattern_type %tuple.elem0.loc8_41.1 [symbolic = %pattern_type.loc8_37 (constants.%pattern_type.08e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_21: <witness> = require_complete_type %HoldsType.loc8_34.1 [symbolic = %require_complete.loc8_21 (constants.%require_complete.514)]
// CHECK:STDOUT:   %require_complete.loc8_38: <witness> = require_complete_type %tuple.elem0.loc8_41.1 [symbolic = %require_complete.loc8_38 (constants.%require_complete.fec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_34.1 (%HoldsType.cc9), %a.param: @F.%tuple.elem0.loc8_41.1 (%tuple.elem0)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc13_25: %tuple.type = tuple_literal (%C.ref)
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc13_26: %tuple.type = converted %.loc13_25, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple) [concrete = constants.%HoldsType.c09]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.c09 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.c09 = class_init (), %.loc13_6.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_6.4: ref %HoldsType.c09 = temporary %.loc13_6.2, %.loc13_6.3
// CHECK:STDOUT:   %.loc13_8.1: ref %HoldsType.c09 = converted %.loc13_6.1, %.loc13_6.4
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: %HoldsType.c09 = bind_value %.loc13_8.1
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C = class_init (), %.loc13_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: ref %C = temporary %.loc13_30.2, %.loc13_30.3
// CHECK:STDOUT:   %.loc13_30.5: ref %C = converted %.loc13_30.1, %.loc13_30.4
// CHECK:STDOUT:   %.loc13_30.6: %C = bind_value %.loc13_30.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.2, %.loc13_30.6)
// CHECK:STDOUT:   %Op.bound.loc13_30: <bound method> = bound_method %.loc13_30.2, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.5d7, @Op.1(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:   %bound_method.loc13_30: <bound method> = bound_method %.loc13_30.2, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_30: %ptr.019 = addr_of %.loc13_30.2
// CHECK:STDOUT:   %no_op.loc13_30: init %empty_tuple.type = call %bound_method.loc13_30(%addr.loc13_30)
// CHECK:STDOUT:   %Op.bound.loc13_6: <bound method> = bound_method %.loc13_6.2, constants.%Op.f20
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.f20, @Op.1(constants.%HoldsType.c09) [concrete = constants.%Op.specific_fn.58d]
// CHECK:STDOUT:   %bound_method.loc13_6: <bound method> = bound_method %.loc13_6.2, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc13_6: %ptr.79a = addr_of %.loc13_6.2
// CHECK:STDOUT:   %no_op.loc13_6: init %empty_tuple.type = call %bound_method.loc13_6(%addr.loc13_6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.6eb) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.6eb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.6eb) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T.6eb
// CHECK:STDOUT:   %HoldsType.loc8_34.1 => constants.%HoldsType.cc9
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.ec6
// CHECK:STDOUT:   %tuple.elem0.loc8_41.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.08e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%tuple
// CHECK:STDOUT:   %HoldsType.loc8_34.1 => constants.%HoldsType.c09
// CHECK:STDOUT:   %pattern_type.loc8_20 => constants.%pattern_type.8e0
// CHECK:STDOUT:   %tuple.elem0.loc8_41.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_21 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc8_38 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %T.08d: %struct_type.t = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f2: type = pattern_type %struct_type.t [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.843: type = class_type @HoldsType, @HoldsType(%T.08d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.aa4: type = pattern_type %HoldsType.843 [symbolic]
// CHECK:STDOUT:   %.20a: type = struct_access %T.08d, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9f0: type = pattern_type %.20a [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b19: <witness> = require_complete_type %HoldsType.843 [symbolic]
// CHECK:STDOUT:   %require_complete.1b9: <witness> = require_complete_type %.20a [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.705: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.705 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.50b: type = pattern_type %HoldsType.705 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.1, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6eb: <specific function> = specific_function %Op.5d7, @Op.1(%C) [concrete]
// CHECK:STDOUT:   %Op.type.8ea: type = fn_type @Op.1, @Destroy.impl(%HoldsType.705) [concrete]
// CHECK:STDOUT:   %Op.180: %Op.type.8ea = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d1: type = ptr_type %HoldsType.705 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.fde: <specific function> = specific_function %Op.180, @Op.1(%HoldsType.705) [concrete]
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
// CHECK:STDOUT:     %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     %T.loc4_17.2: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.08d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_22 (%pattern_type.aa4) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_22 (%pattern_type.aa4) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_39 (%pattern_type.9f0) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_39 (%pattern_type.9f0) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     %T.loc8_6.2: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.08d)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_36.1 (%HoldsType.843) = value_param call_param0
// CHECK:STDOUT:     %.loc8_36: type = splice_block %HoldsType.loc8_36.2 [symbolic = %HoldsType.loc8_36.1 (constants.%HoldsType.843)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_35: %struct_type.t = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.08d)]
// CHECK:STDOUT:       %HoldsType.loc8_36.2: type = class_type @HoldsType, @HoldsType(constants.%T.08d) [symbolic = %HoldsType.loc8_36.1 (constants.%HoldsType.843)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_36.1 (%HoldsType.843) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_43.1 (%.20a) = value_param call_param1
// CHECK:STDOUT:     %.loc8_43.2: type = splice_block %.loc8_43.3 [symbolic = %.loc8_43.1 (constants.%.20a)] {
// CHECK:STDOUT:       %T.ref.loc8_42: %struct_type.t = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.08d)]
// CHECK:STDOUT:       %.loc8_43.3: type = struct_access %T.ref.loc8_42, element0 [symbolic = %.loc8_43.1 (constants.%.20a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_43.1 (%.20a) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %struct_type.t) {
// CHECK:STDOUT:   %T.loc4_17.1: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.08d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.843
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %struct_type.t) {
// CHECK:STDOUT:   %T.loc8_6.1: %struct_type.t = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.08d)]
// CHECK:STDOUT:   %HoldsType.loc8_36.1: type = class_type @HoldsType, @HoldsType(%T.loc8_6.1) [symbolic = %HoldsType.loc8_36.1 (constants.%HoldsType.843)]
// CHECK:STDOUT:   %pattern_type.loc8_22: type = pattern_type %HoldsType.loc8_36.1 [symbolic = %pattern_type.loc8_22 (constants.%pattern_type.aa4)]
// CHECK:STDOUT:   %.loc8_43.1: type = struct_access %T.loc8_6.1, element0 [symbolic = %.loc8_43.1 (constants.%.20a)]
// CHECK:STDOUT:   %pattern_type.loc8_39: type = pattern_type %.loc8_43.1 [symbolic = %pattern_type.loc8_39 (constants.%pattern_type.9f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_23: <witness> = require_complete_type %HoldsType.loc8_36.1 [symbolic = %require_complete.loc8_23 (constants.%require_complete.b19)]
// CHECK:STDOUT:   %require_complete.loc8_40: <witness> = require_complete_type %.loc8_43.1 [symbolic = %require_complete.loc8_40 (constants.%require_complete.1b9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_36.1 (%HoldsType.843), %a.param: @F.%.loc8_43.1 (%.20a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc13_28: %struct_type.t = struct_literal (%C.ref)
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc13_29: %struct_type.t = converted %.loc13_28, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.705]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.705 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.705 = class_init (), %.loc13_6.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_6.4: ref %HoldsType.705 = temporary %.loc13_6.2, %.loc13_6.3
// CHECK:STDOUT:   %.loc13_8.1: ref %HoldsType.705 = converted %.loc13_6.1, %.loc13_6.4
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: %HoldsType.705 = bind_value %.loc13_8.1
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C = class_init (), %.loc13_33.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: ref %C = temporary %.loc13_33.2, %.loc13_33.3
// CHECK:STDOUT:   %.loc13_33.5: ref %C = converted %.loc13_33.1, %.loc13_33.4
// CHECK:STDOUT:   %.loc13_33.6: %C = bind_value %.loc13_33.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.2, %.loc13_33.6)
// CHECK:STDOUT:   %Op.bound.loc13_33: <bound method> = bound_method %.loc13_33.2, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.5d7, @Op.1(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:   %bound_method.loc13_33: <bound method> = bound_method %.loc13_33.2, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_33: %ptr.019 = addr_of %.loc13_33.2
// CHECK:STDOUT:   %no_op.loc13_33: init %empty_tuple.type = call %bound_method.loc13_33(%addr.loc13_33)
// CHECK:STDOUT:   %Op.bound.loc13_6: <bound method> = bound_method %.loc13_6.2, constants.%Op.180
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.180, @Op.1(constants.%HoldsType.705) [concrete = constants.%Op.specific_fn.fde]
// CHECK:STDOUT:   %bound_method.loc13_6: <bound method> = bound_method %.loc13_6.2, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc13_6: %ptr.5d1 = addr_of %.loc13_6.2
// CHECK:STDOUT:   %no_op.loc13_6: init %empty_tuple.type = call %bound_method.loc13_6(%addr.loc13_6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.08d) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.08d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.08d) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T.08d
// CHECK:STDOUT:   %HoldsType.loc8_36.1 => constants.%HoldsType.843
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.aa4
// CHECK:STDOUT:   %.loc8_43.1 => constants.%.20a
// CHECK:STDOUT:   %pattern_type.loc8_39 => constants.%pattern_type.9f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%struct) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%struct
// CHECK:STDOUT:   %HoldsType.loc8_36.1 => constants.%HoldsType.705
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.50b
// CHECK:STDOUT:   %.loc8_43.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_39 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_23 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc8_40 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_class_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, type [concrete]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %complete_type.509: <witness> = complete_type_witness %struct_type.t [concrete]
// CHECK:STDOUT:   %T.0de: %Class = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.f95cf2.1: type = class_type @HoldsType, @HoldsType(%T.0de) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.937: type = pattern_type %HoldsType.f95cf2.1 [symbolic]
// CHECK:STDOUT:   %.2fe: ref type = class_element_access %T.0de, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.8fa644.1: <witness> = require_complete_type %HoldsType.f95cf2.1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %c: %Class = bind_symbolic_name c, 0 [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.f95cf2.2: type = class_type @HoldsType, @HoldsType(%c) [symbolic]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.f95cf2.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %ptr.deb: type = ptr_type %HoldsType.f95cf2.2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %HoldsType.f95cf2.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.088: %Destroy.type = facet_value %HoldsType.f95cf2.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.da6: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.088 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.da6 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.2(%Destroy.facet.088) [symbolic]
// CHECK:STDOUT:   %Op.type.7de: type = fn_type @Op.1, @Destroy.impl(%Class) [concrete]
// CHECK:STDOUT:   %Op.d64: %Op.type.7de = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.d64, @Op.1(%Class) [concrete]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .HoldsType = %HoldsType.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %HoldsType.decl: %HoldsType.type = class_decl @HoldsType [concrete = constants.%HoldsType.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.761 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %T.loc8_17.2: %Class = bind_symbolic_name T, 0 [symbolic = %T.loc8_17.1 (constants.%T.0de)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.761 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.937) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.937) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %T.loc21_6.2: %Class = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.1 (constants.%T.0de)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc21_31.1 (%HoldsType.f95cf2.1) = value_param call_param0
// CHECK:STDOUT:     %.loc21_31: type = splice_block %HoldsType.loc21_31.2 [symbolic = %HoldsType.loc21_31.1 (constants.%HoldsType.f95cf2.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc21_30: %Class = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T.0de)]
// CHECK:STDOUT:       %HoldsType.loc21_31.2: type = class_type @HoldsType, @HoldsType(constants.%T.0de) [symbolic = %HoldsType.loc21_31.1 (constants.%HoldsType.f95cf2.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc21_31.1 (%HoldsType.f95cf2.1) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc21_37: %Class = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T.0de)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:       %.loc21_38.2: ref type = class_element_access %T.ref.loc21_37, element0 [symbolic = %.loc21_38.1 (constants.%.2fe)]
// CHECK:STDOUT:       %.loc21_38.3: type = bind_value %.loc21_38.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc5: %Class.elem = field_decl t, element0 [concrete]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.t [concrete = constants.%complete_type.509]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .t = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc8_17.2: %Class) {
// CHECK:STDOUT:   %T.loc8_17.1: %Class = bind_symbolic_name T, 0 [symbolic = %T.loc8_17.1 (constants.%T.0de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.f95cf2.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc21_6.2: %Class) {
// CHECK:STDOUT:   %T.loc21_6.1: %Class = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.1 (constants.%T.0de)]
// CHECK:STDOUT:   %HoldsType.loc21_31.1: type = class_type @HoldsType, @HoldsType(%T.loc21_6.1) [symbolic = %HoldsType.loc21_31.1 (constants.%HoldsType.f95cf2.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc21_31.1 [symbolic = %pattern_type (constants.%pattern_type.937)]
// CHECK:STDOUT:   %.loc21_38.1: ref type = class_element_access %T.loc21_6.1, element0 [symbolic = %.loc21_38.1 (constants.%.2fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc21_31.1 [symbolic = %require_complete (constants.%require_complete.8fa644.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc21_31.1 (%HoldsType.f95cf2.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = symbolic_binding_pattern c, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc26_26.1: %struct_type.t = struct_literal (%C.ref)
// CHECK:STDOUT:   %Class.ref.loc26_31: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc26_26.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc26_26.3: ref type = class_element_access %.loc26_26.2, element0
// CHECK:STDOUT:   %.loc26_26.4: init type = initialize_from %C.ref to %.loc26_26.3 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc26_26.5: init %Class = class_init (%.loc26_26.4), %.loc26_26.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc26_26.6: ref %Class = temporary %.loc26_26.2, %.loc26_26.5
// CHECK:STDOUT:   %.loc26_28.1: ref %Class = converted %.loc26_26.1, %.loc26_26.6
// CHECK:STDOUT:   %Class.ref.loc26_11: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc26_28.2: %Class = bind_value %.loc26_28.1
// CHECK:STDOUT:   %c: %Class = bind_symbolic_name c, 0, %.loc26_28.2 [symbolic = constants.%c]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc27_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c [symbolic = constants.%c]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%c) [symbolic = constants.%HoldsType.f95cf2.2]
// CHECK:STDOUT:   %.loc27_6.2: ref %HoldsType.f95cf2.2 = temporary_storage
// CHECK:STDOUT:   %.loc27_6.3: init %HoldsType.f95cf2.2 = class_init (), %.loc27_6.2 [symbolic = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc27_6.4: ref %HoldsType.f95cf2.2 = temporary %.loc27_6.2, %.loc27_6.3
// CHECK:STDOUT:   %.loc27_8: ref %HoldsType.f95cf2.2 = converted %.loc27_6.1, %.loc27_6.4
// CHECK:STDOUT:   %.loc27_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %impl.elem0: %.da6 = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:   %bound_method.loc27_6.1: <bound method> = bound_method %.loc27_6.2, %impl.elem0
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.2(constants.%Destroy.facet.088) [symbolic = constants.%specific_impl_fn]
// CHECK:STDOUT:   %bound_method.loc27_6.2: <bound method> = bound_method %.loc27_6.2, %specific_impl_fn
// CHECK:STDOUT:   %addr.loc27: %ptr.deb = addr_of %.loc27_6.2
// CHECK:STDOUT:   %.loc27_6.5: init %empty_tuple.type = call %bound_method.loc27_6.2(%addr.loc27)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc26_26.2, constants.%Op.d64
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.d64, @Op.1(constants.%Class) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %.loc26_26.2, %Op.specific_fn
// CHECK:STDOUT:   %addr.loc26: %ptr.e71 = addr_of %.loc26_26.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc26(%addr.loc26)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.0de) {
// CHECK:STDOUT:   %T.loc8_17.1 => constants.%T.0de
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.0de) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T.0de
// CHECK:STDOUT:   %HoldsType.loc21_31.1 => constants.%HoldsType.f95cf2.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.937
// CHECK:STDOUT:   %.loc21_38.1 => constants.%.2fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%c) {
// CHECK:STDOUT:   %T.loc8_17.1 => constants.%c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, type [concrete]
// CHECK:STDOUT:   %T.eb6: %array_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dcb: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(%T.eb6) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.ea3: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %pattern_type.142: type = pattern_type %HoldsType [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.640: <witness> = require_complete_type %HoldsType [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.8a7: type = fn_type @Op.2, @Destroy.impl(%array_type) [concrete]
// CHECK:STDOUT:   %Op.246: %Op.type.8a7 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.246, @Op.2(%array_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
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
// CHECK:STDOUT:     %.loc4: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, type [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: %array_type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.eb6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.dcb = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.142) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.142) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, type [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_6.2: %array_type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.1 (constants.%T.eb6)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc12_40.1 (%HoldsType) = value_param call_param0
// CHECK:STDOUT:     %.loc12_40: type = splice_block %HoldsType.loc12_40.2 [symbolic = %HoldsType.loc12_40.1 (constants.%HoldsType)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc12_39: %array_type = name_ref T, %T.loc12_6.2 [symbolic = %T.loc12_6.1 (constants.%T.eb6)]
// CHECK:STDOUT:       %HoldsType.loc12_40.2: type = class_type @HoldsType, @HoldsType(constants.%T.eb6) [symbolic = %HoldsType.loc12_40.1 (constants.%HoldsType)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc12_40.1 (%HoldsType) = bind_name x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc12_46: %array_type = name_ref T, %T.loc12_6.2 [symbolic = %T.loc12_6.1 (constants.%T.eb6)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc12_48.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_48.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %bound_method.loc12_48.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_48.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_48.2: %i32 = converted %int_0, %.loc12_48.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_49.1: ref %array_type = value_as_ref %T.ref.loc12_46
// CHECK:STDOUT:       %.loc12_49.2: ref type = array_index %.loc12_49.1, %.loc12_48.2
// CHECK:STDOUT:       %.loc12_49.3: type = bind_value %.loc12_49.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %array_type) {
// CHECK:STDOUT:   %T.loc4_17.1: %array_type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.eb6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_6.2: %array_type) {
// CHECK:STDOUT:   %T.loc12_6.1: %array_type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.1 (constants.%T.eb6)]
// CHECK:STDOUT:   %HoldsType.loc12_40.1: type = class_type @HoldsType, @HoldsType(%T.loc12_6.1) [symbolic = %HoldsType.loc12_40.1 (constants.%HoldsType)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc12_40.1 [symbolic = %pattern_type (constants.%pattern_type.142)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc12_40.1 [symbolic = %require_complete (constants.%require_complete.640)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc12_40.1 (%HoldsType), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc24_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_25.1: %tuple.type = tuple_literal (%C.ref)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, type [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc24_25.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc24_25.3: ref type = array_index %.loc24_25.2, %int_0
// CHECK:STDOUT:   %.loc24_25.4: init type = initialize_from %C.ref to %.loc24_25.3 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_25.5: init %array_type = array_init (%.loc24_25.4) to %.loc24_25.2 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_27.1: init %array_type = converted %.loc24_25.1, %.loc24_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_27.2: ref %array_type = temporary %.loc24_25.2, %.loc24_27.1
// CHECK:STDOUT:   %.loc24_27.3: %array_type = bind_value %.loc24_27.2
// CHECK:STDOUT:   %.loc24_48: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc24_25.2, constants.%Op.246
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.246, @Op.2(constants.%array_type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_25.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ea3 = addr_of %.loc24_25.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.eb6) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.eb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.eb6) {
// CHECK:STDOUT:   %T.loc12_6.1 => constants.%T.eb6
// CHECK:STDOUT:   %HoldsType.loc12_40.1 => constants.%HoldsType
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.142
// CHECK:STDOUT: }
// CHECK:STDOUT:
