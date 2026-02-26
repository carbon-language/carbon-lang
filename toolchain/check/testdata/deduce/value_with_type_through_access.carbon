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
  // CHECK:STDERR: fail_todo_class_access.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G({.t = C});
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_class_access.carbon:[[@LINE-8]]:6: note: initializing generic parameter `c` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G(c:! Class) {
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.0d2: %tuple.type = tuple_value (type) [concrete]
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.f6a: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.17d: type = pattern_type %HoldsType.f6a [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.7bf: <witness> = require_complete_type %HoldsType.f6a [symbolic]
// CHECK:STDOUT:   %require_complete.e7a: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.1f1: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.a31: type = class_type @HoldsType, @HoldsType(%tuple.1f1) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.a31 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a646: type = pattern_type %HoldsType.a31 [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.1f1) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc13_30 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc13_6 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.loc4_17.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f1e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_27 (%pattern_type.17d) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_27 (%pattern_type.17d) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_51 (%pattern_type.e66) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_51 (%pattern_type.e66) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_11: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc8_17.2: %tuple.type = tuple_literal (%.loc8_11) [concrete = constants.%tuple.0d2]
// CHECK:STDOUT:       %.loc8_17.3: type = converted %.loc8_17.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.f6a) = value_param call_param0
// CHECK:STDOUT:     %.loc8_41: type = splice_block %HoldsType.loc8_41.2 [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.f6a)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_40: %tuple.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_41.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.f6a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_41.1 (%HoldsType.f6a) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_55: type = splice_block %tuple.elem0.loc8_55.2 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_54: %tuple.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc8_55.2: type = tuple_access %T.ref.loc8_54, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc4_17.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.f6a
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %tuple.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_41.1: type = class_type @HoldsType, @HoldsType(%T.loc8_6.1) [symbolic = %HoldsType.loc8_41.1 (constants.%HoldsType.f6a)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %HoldsType.loc8_41.1 [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.17d)]
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1: type = tuple_access %T.loc8_6.1, element0 [symbolic = %tuple.elem0.loc8_55.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_51: type = pattern_type %tuple.elem0.loc8_55.1 [symbolic = %pattern_type.loc8_51 (constants.%pattern_type.e66)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_28: <witness> = require_complete_type %HoldsType.loc8_41.1 [symbolic = %require_complete.loc8_28 (constants.%require_complete.7bf)]
// CHECK:STDOUT:   %require_complete.loc8_52: <witness> = require_complete_type %tuple.elem0.loc8_55.1 [symbolic = %require_complete.loc8_52 (constants.%require_complete.e7a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_41.1 (%HoldsType.f6a), %a.param: @F.%tuple.elem0.loc8_55.1 (%tuple.elem0)) {
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
// CHECK:STDOUT:   %.loc13_25: %tuple.type = tuple_literal (%C.ref) [concrete = constants.%tuple.1f1]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple.1f1]
// CHECK:STDOUT:   %.loc13_26: %tuple.type = converted %.loc13_25, %tuple [concrete = constants.%tuple.1f1]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.1f1) [concrete = constants.%HoldsType.a31]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.a31 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.a31 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_6.4: ref %HoldsType.a31 = temporary %.loc13_6.2, %.loc13_6.3
// CHECK:STDOUT:   %.loc13_8.1: ref %HoldsType.a31 = converted %.loc13_6.1, %.loc13_6.4
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.1f1) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: %HoldsType.a31 = acquire_value %.loc13_8.1
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C to %.loc13_30.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: ref %C = temporary %.loc13_30.2, %.loc13_30.3
// CHECK:STDOUT:   %.loc13_30.5: ref %C = converted %.loc13_30.1, %.loc13_30.4
// CHECK:STDOUT:   %.loc13_30.6: %C = acquire_value %.loc13_30.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.2, %.loc13_30.6)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_30: <bound method> = bound_method %.loc13_30.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc13_30: init %empty_tuple.type = call %DestroyOp.bound.loc13_30(%.loc13_30.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_6: <bound method> = bound_method %.loc13_6.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc13_6: init %empty_tuple.type = call %DestroyOp.bound.loc13_6(%.loc13_6.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13_30(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13_6(%self.param: ref %HoldsType.a31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.f6a
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.17d
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_51 => constants.%pattern_type.e66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.1f1) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%tuple.1f1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.1f1) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%tuple.1f1
// CHECK:STDOUT:   %HoldsType.loc8_41.1 => constants.%HoldsType.a31
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.a646
// CHECK:STDOUT:   %tuple.elem0.loc8_55.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_51 => constants.%pattern_type.7c7
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
// CHECK:STDOUT:   %T: %struct_type.t = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f2: type = pattern_type %struct_type.t [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.cac: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.2de: type = pattern_type %HoldsType.cac [symbolic]
// CHECK:STDOUT:   %.424: type = struct_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.92c: type = pattern_type %.424 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.276: <witness> = require_complete_type %HoldsType.cac [symbolic]
// CHECK:STDOUT:   %require_complete.ba0: <witness> = require_complete_type %.424 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.673: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.673 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4e6: type = pattern_type %HoldsType.673 [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc13_33 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc13_6 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.loc4_17.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_29 (%pattern_type.2de) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_29 (%pattern_type.2de) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc8_53 (%pattern_type.92c) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc8_53 (%pattern_type.92c) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_15: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.cac) = value_param call_param0
// CHECK:STDOUT:     %.loc8_43: type = splice_block %HoldsType.loc8_43.2 [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.cac)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_42: %struct_type.t = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_43.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.cac)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_43.1 (%HoldsType.cac) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_57.1 (%.424) = value_param call_param1
// CHECK:STDOUT:     %.loc8_57.2: type = splice_block %.loc8_57.3 [symbolic = %.loc8_57.1 (constants.%.424)] {
// CHECK:STDOUT:       %T.ref.loc8_56: %struct_type.t = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_57.3: type = struct_access %T.ref.loc8_56, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_57.1 (%.424) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %struct_type.t) {
// CHECK:STDOUT:   %T.loc4_17.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.cac
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %struct_type.t) {
// CHECK:STDOUT:   %T.loc8_6.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_43.1: type = class_type @HoldsType, @HoldsType(%T.loc8_6.1) [symbolic = %HoldsType.loc8_43.1 (constants.%HoldsType.cac)]
// CHECK:STDOUT:   %pattern_type.loc8_29: type = pattern_type %HoldsType.loc8_43.1 [symbolic = %pattern_type.loc8_29 (constants.%pattern_type.2de)]
// CHECK:STDOUT:   %.loc8_57.1: type = struct_access %T.loc8_6.1, element0 [symbolic = %.loc8_57.1 (constants.%.424)]
// CHECK:STDOUT:   %pattern_type.loc8_53: type = pattern_type %.loc8_57.1 [symbolic = %pattern_type.loc8_53 (constants.%pattern_type.92c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_30: <witness> = require_complete_type %HoldsType.loc8_43.1 [symbolic = %require_complete.loc8_30 (constants.%require_complete.276)]
// CHECK:STDOUT:   %require_complete.loc8_54: <witness> = require_complete_type %.loc8_57.1 [symbolic = %require_complete.loc8_54 (constants.%require_complete.ba0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_43.1 (%HoldsType.cac), %a.param: @F.%.loc8_57.1 (%.424)) {
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
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.673]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.673 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.673 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_6.4: ref %HoldsType.673 = temporary %.loc13_6.2, %.loc13_6.3
// CHECK:STDOUT:   %.loc13_8.1: ref %HoldsType.673 = converted %.loc13_6.1, %.loc13_6.4
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: %HoldsType.673 = acquire_value %.loc13_8.1
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C to %.loc13_33.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: ref %C = temporary %.loc13_33.2, %.loc13_33.3
// CHECK:STDOUT:   %.loc13_33.5: ref %C = converted %.loc13_33.1, %.loc13_33.4
// CHECK:STDOUT:   %.loc13_33.6: %C = acquire_value %.loc13_33.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.2, %.loc13_33.6)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_33: <bound method> = bound_method %.loc13_33.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc13_33: init %empty_tuple.type = call %DestroyOp.bound.loc13_33(%.loc13_33.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_6: <bound method> = bound_method %.loc13_6.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc13_6: init %empty_tuple.type = call %DestroyOp.bound.loc13_6(%.loc13_6.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13_33(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13_6(%self.param: ref %HoldsType.673) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.cac
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.2de
// CHECK:STDOUT:   %.loc8_57.1 => constants.%.424
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.92c
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
// CHECK:STDOUT:   %HoldsType.loc8_43.1 => constants.%HoldsType.673
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.4e6
// CHECK:STDOUT:   %.loc8_57.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.7c7
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
// CHECK:STDOUT:   %T.d7d: %Class = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.904: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.47b504.1: type = class_type @HoldsType, @HoldsType(%T.d7d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.3b8c03.1: type = pattern_type %HoldsType.47b504.1 [symbolic]
// CHECK:STDOUT:   %.208: ref type = class_element_access %T.d7d, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9b8c71.1: <witness> = require_complete_type %HoldsType.47b504.1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %c: %Class = symbolic_binding c, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.47b504.2: type = class_type @HoldsType, @HoldsType(%c) [symbolic]
// CHECK:STDOUT:   %require_complete.9b8c71.2: <witness> = require_complete_type %HoldsType.47b504.2 [symbolic]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.47b504.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %HoldsType.47b504.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.278: %Destroy.type = facet_value %HoldsType.47b504.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.055: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.278) [symbolic]
// CHECK:STDOUT:   %.b9b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.055, %Destroy.facet.278 [symbolic]
// CHECK:STDOUT:   %impl.elem0.909: %.b9b = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2d3: <specific function> = specific_impl_function %impl.elem0.909, @Destroy.WithSelf.Op(%Destroy.facet.278) [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.904 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_17.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_17.1 (constants.%T.d7d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.904 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.3b8c03.1) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.3b8c03.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: <error> = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_10: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_6.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T.d7d)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.47b504.1) = value_param call_param0
// CHECK:STDOUT:     %.loc21_38: type = splice_block %HoldsType.loc21_38.2 [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.47b504.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc21_37: %Class = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T.d7d)]
// CHECK:STDOUT:       %HoldsType.loc21_38.2: type = class_type @HoldsType, @HoldsType(constants.%T.d7d) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.47b504.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc21_38.1 (%HoldsType.47b504.1) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc21_51: %Class = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T.d7d)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:       %.loc21_52.2: ref type = class_element_access %T.ref.loc21_51, element0 [symbolic = %.loc21_52.1 (constants.%.208)]
// CHECK:STDOUT:       %.loc21_52.3: type = acquire_value %.loc21_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %pattern_type.904 = symbolic_binding_pattern c, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc25_6.2: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_6.1 (constants.%c)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc5_10: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc5_8: %Class.elem = field_decl t, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.t [concrete = constants.%complete_type.509]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .t = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc8_17.2: %Class) {
// CHECK:STDOUT:   %T.loc8_17.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_17.1 (constants.%T.d7d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.47b504.1
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
// CHECK:STDOUT: generic fn @F(%T.loc21_6.2: %Class) {
// CHECK:STDOUT:   %T.loc21_6.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T.d7d)]
// CHECK:STDOUT:   %HoldsType.loc21_38.1: type = class_type @HoldsType, @HoldsType(%T.loc21_6.1) [symbolic = %HoldsType.loc21_38.1 (constants.%HoldsType.47b504.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc21_38.1 [symbolic = %pattern_type (constants.%pattern_type.3b8c03.1)]
// CHECK:STDOUT:   %.loc21_52.1: ref type = class_element_access %T.loc21_6.1, element0 [symbolic = %.loc21_52.1 (constants.%.208)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc21_38.1 [symbolic = %require_complete (constants.%require_complete.9b8c71.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc21_38.1 (%HoldsType.47b504.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%c.loc25_6.2: %Class) {
// CHECK:STDOUT:   %c.loc25_6.1: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_6.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2: type = class_type @HoldsType, @HoldsType(%c.loc25_6.1) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.47b504.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc26_22.2 [symbolic = %require_complete (constants.%require_complete.9b8c71.2)]
// CHECK:STDOUT:   %HoldsType.val: @G.%HoldsType.loc26_22.2 (%HoldsType.47b504.2) = struct_value () [symbolic = %HoldsType.val (constants.%HoldsType.val)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %HoldsType.loc26_22.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %HoldsType.loc26_22.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.278)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.055)]
// CHECK:STDOUT:   %.loc26_6.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc26_6.5 (constants.%.b9b)]
// CHECK:STDOUT:   %impl.elem0.loc26_6.2: @G.%.loc26_6.5 (%.b9b) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_6.2 (constants.%impl.elem0.909)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_6.2: <specific function> = specific_impl_function %impl.elem0.loc26_6.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc26_6.2 (constants.%specific_impl_fn.2d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc26_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:     %c.ref: %Class = name_ref c, %c.loc25_6.2 [symbolic = %c.loc25_6.1 (constants.%c)]
// CHECK:STDOUT:     %HoldsType.loc26_22.1: type = class_type @HoldsType, @HoldsType(constants.%c) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.47b504.2)]
// CHECK:STDOUT:     %.loc26_6.2: ref @G.%HoldsType.loc26_22.2 (%HoldsType.47b504.2) = temporary_storage
// CHECK:STDOUT:     %.loc26_6.3: init @G.%HoldsType.loc26_22.2 (%HoldsType.47b504.2) to %.loc26_6.2 = class_init () [symbolic = %HoldsType.val (constants.%HoldsType.val)]
// CHECK:STDOUT:     %.loc26_6.4: ref @G.%HoldsType.loc26_22.2 (%HoldsType.47b504.2) = temporary %.loc26_6.2, %.loc26_6.3
// CHECK:STDOUT:     %.loc26_8: ref @G.%HoldsType.loc26_22.2 (%HoldsType.47b504.2) = converted %.loc26_6.1, %.loc26_6.4
// CHECK:STDOUT:     %.loc26_26: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %impl.elem0.loc26_6.1: @G.%.loc26_6.5 (%.b9b) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_6.2 (constants.%impl.elem0.909)]
// CHECK:STDOUT:     %bound_method.loc26_6.1: <bound method> = bound_method %.loc26_6.4, %impl.elem0.loc26_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_6.1: <specific function> = specific_impl_function %impl.elem0.loc26_6.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.278) [symbolic = %specific_impl_fn.loc26_6.2 (constants.%specific_impl_fn.2d3)]
// CHECK:STDOUT:     %bound_method.loc26_6.2: <bound method> = bound_method %.loc26_6.4, %specific_impl_fn.loc26_6.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc26_6.2(%.loc26_6.4)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc37_12.1: %struct_type.t = struct_literal (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc37_12.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc37_12.3: ref type = class_element_access %.loc37_12.2, element0
// CHECK:STDOUT:   %.loc37_12.4: init type to %.loc37_12.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc37_12.5: init %Class to %.loc37_12.2 = class_init (%.loc37_12.4) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc37_12.6: ref %Class = temporary %.loc37_12.2, %.loc37_12.5
// CHECK:STDOUT:   %.loc37_13.1: ref %Class = converted %.loc37_12.1, %.loc37_12.6
// CHECK:STDOUT:   %.loc37_13.2: %Class = acquire_value %.loc37_13.1
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc37_12.6, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc37_12.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %Class) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.d7d) {
// CHECK:STDOUT:   %T.loc8_17.1 => constants.%T.d7d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.d7d) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T.d7d
// CHECK:STDOUT:   %HoldsType.loc21_38.1 => constants.%HoldsType.47b504.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3b8c03.1
// CHECK:STDOUT:   %.loc21_52.1 => constants.%.208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%c) {
// CHECK:STDOUT:   %c.loc25_6.1 => constants.%c
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, type [concrete]
// CHECK:STDOUT:   %T.9b7: %array_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dcb: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(%T.9b7) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.342: type = pattern_type %HoldsType [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.bd5: <witness> = require_complete_type %HoldsType [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.1f1: %tuple.type.85c = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.loc4_17.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.9b7)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.dcb = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.342) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.342) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: <error> = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_16: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc12_16 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_6.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_6.1 (constants.%T.9b7)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc12_47.1 (%HoldsType) = value_param call_param0
// CHECK:STDOUT:     %.loc12_47: type = splice_block %HoldsType.loc12_47.2 [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc12_46: %array_type = name_ref T, %T.loc12_6.2 [symbolic = %T.loc12_6.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %HoldsType.loc12_47.2: type = class_type @HoldsType, @HoldsType(constants.%T.9b7) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc12_47.1 (%HoldsType) = value_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc12_60: %array_type = name_ref T, %T.loc12_6.2 [symbolic = %T.loc12_6.1 (constants.%T.9b7)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:       %bound_method.loc12_62.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_62.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_62.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_62.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_62.2: %i32 = converted %int_0, %.loc12_62.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc12_63.1: ref %array_type = value_as_ref %T.ref.loc12_60
// CHECK:STDOUT:       %.loc12_63.2: ref type = array_index %.loc12_63.1, %.loc12_62.2
// CHECK:STDOUT:       %.loc12_63.3: type = acquire_value %.loc12_63.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_17.2: %array_type) {
// CHECK:STDOUT:   %T.loc4_17.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.9b7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType
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
// CHECK:STDOUT: generic fn @F(%T.loc12_6.2: %array_type) {
// CHECK:STDOUT:   %T.loc12_6.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_6.1 (constants.%T.9b7)]
// CHECK:STDOUT:   %HoldsType.loc12_47.1: type = class_type @HoldsType, @HoldsType(%T.loc12_6.1) [symbolic = %HoldsType.loc12_47.1 (constants.%HoldsType)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc12_47.1 [symbolic = %pattern_type (constants.%pattern_type.342)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc12_47.1 [symbolic = %require_complete (constants.%require_complete.bd5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc12_47.1 (%HoldsType), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc24_6: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_25.1: %tuple.type.85c = tuple_literal (%C.ref) [concrete = constants.%tuple.1f1]
// CHECK:STDOUT:   %.loc24_36: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %.loc24_36 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_25.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc24_25.3: ref type = array_index %.loc24_25.2, %int_0
// CHECK:STDOUT:   %.loc24_25.4: init type to %.loc24_25.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc24_25.5: init %array_type to %.loc24_25.2 = array_init (%.loc24_25.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_27.1: init %array_type = converted %.loc24_25.1, %.loc24_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc24_27.2: ref %array_type = temporary %.loc24_25.2, %.loc24_27.1
// CHECK:STDOUT:   %.loc24_27.3: %array_type = acquire_value %.loc24_27.2
// CHECK:STDOUT:   %.loc24_48: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc24_27.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc24_27.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.9b7) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.9b7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.9b7) {
// CHECK:STDOUT:   %T.loc12_6.1 => constants.%T.9b7
// CHECK:STDOUT:   %HoldsType.loc12_47.1 => constants.%HoldsType
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.342
// CHECK:STDOUT: }
// CHECK:STDOUT:
