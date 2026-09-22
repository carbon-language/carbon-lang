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

class HoldsType(T: (type, )) {}

// `a` is received as a value, and has a type that requires going through
// TupleAccess. Building the value representation must handle this indirection.
fn F[T: (type, )](unused x: HoldsType(T), unused a: T.0) {}

class C {}

fn G() {
  F({} as HoldsType((C, )), {});
}

// --- struct_access.carbon

library "[[@TEST_NAME]]";

class HoldsType(T: {.t: type}) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
fn F[T: {.t: type}](unused x: HoldsType(T), unused a: T.t) {}

class C {}

fn G() {
  F({} as HoldsType({.t = C}), {});
}

// --- fail_todo_class_access.carbon

library "[[@TEST_NAME]]";

class Class {
  var t: type;
}

class HoldsType(T: Class) {}

// `a` is received as a value, and has a type that requires going through
// StructAccess. Building the value representation must handle this indirection.
//
// TODO: The `Class` value in `T` is a comptime constant and this expression
// should be able to eval to a type. We would expect the `.` in `T.t` to be a
// ClassElementAccess but it is an AcquireValue, which has a runtime value.
//
// CHECK:STDERR: fail_todo_class_access.carbon:[[@LINE+4]]:50: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T: Class](unused x: HoldsType(T), unused a: T.t) {}
// CHECK:STDERR:                                                  ^~~
// CHECK:STDERR:
fn F[T: Class](unused x: HoldsType(T), unused a: T.t) {}

class C {}

fn G(generic c: Class) {
  F({} as HoldsType(c), {});
}

fn H() {
  G({.t = C});
}

// --- fail_todo_array_index.carbon

library "[[@TEST_NAME]]";

class HoldsType(T: array(type, 1)) {}

// `a` is received as a value, and has a type that requires going through
// ArrayIndex. Building the value representation must handle this indirection.
// CHECK:STDERR: fail_todo_array_index.carbon:[[@LINE+4]]:59: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[T: array(type, 1)](unused x: HoldsType(T), unused a: T[0]) {}
// CHECK:STDERR:                                                           ^~~~
// CHECK:STDERR:
fn F[T: array(type, 1)](unused x: HoldsType(T), unused a: T[0]) {}

class C {}

fn G() {
  F({} as HoldsType((C, ) as array(type, 1)), {});
}



// CHECK:STDOUT: --- tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type = tuple_value (type) [concrete]
// CHECK:STDOUT:   %pattern_type.ddf: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.4fb: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.ec9: type = pattern_type %HoldsType.4fb [symbolic]
// CHECK:STDOUT:   %x.param_patt.741: %pattern_type.ec9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.991: %pattern_type.ec9 = wrapper_binding_pattern x, %x.param_patt.741 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.251: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %a.param_patt.46f: %pattern_type.251 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.634: %pattern_type.251 = wrapper_binding_pattern a, %a.param_patt.46f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4d4: <witness> = require_complete_type %HoldsType.4fb [symbolic]
// CHECK:STDOUT:   %require_complete.3a8: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.9e1: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.210: type = class_type @HoldsType, @HoldsType(%tuple.9e1) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.210 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2e7: type = pattern_type %HoldsType.210 [concrete]
// CHECK:STDOUT:   %x.param_patt.261: %pattern_type.2e7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c43: %pattern_type.2e7 = wrapper_binding_pattern x, %x.param_patt.261 [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.4cc: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.9e1) [concrete]
// CHECK:STDOUT:   %.533: ref %HoldsType.210 = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc13_30.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound.1a9: <bound method> = bound_method %.115, %Destroy.WithSelf.Op.403171.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound.eeb: <bound method> = bound_method %.533, %Destroy.WithSelf.Op.403171.3 [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_21: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc4_27.2: %tuple.type = tuple_literal (%.loc4_21) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc4_27.3: type = converted %.loc4_27.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_27.1: @F.%pattern_type.loc8_27 (%pattern_type.ec9) = value_param_pattern [symbolic = %x.param_patt.loc8_27.2 (constants.%x.param_patt.741)]
// CHECK:STDOUT:     %x.patt.loc8_27.1: @F.%pattern_type.loc8_27 (%pattern_type.ec9) = wrapper_binding_pattern x, %x.param_patt.loc8_27.1 [symbolic = %x.patt.loc8_27.2 (constants.%x.patt.991)]
// CHECK:STDOUT:     %a.param_patt.loc8_51.1: @F.%pattern_type.loc8_51 (%pattern_type.251) = value_param_pattern [symbolic = %a.param_patt.loc8_51.2 (constants.%a.param_patt.46f)]
// CHECK:STDOUT:     %a.patt.loc8_51.1: @F.%pattern_type.loc8_51 (%pattern_type.251) = wrapper_binding_pattern a, %a.param_patt.loc8_51.1 [symbolic = %a.patt.loc8_51.2 (constants.%a.patt.634)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_16.1: type = splice_block %.loc8_16.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_10: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc8_16.2: %tuple.type = tuple_literal (%.loc8_10) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc8_16.3: type = converted %.loc8_16.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_40.1 (%HoldsType.4fb) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40: type = splice_block %HoldsType.loc8_40.2 [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.4fb)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_39: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_40.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.4fb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_40.1 (%HoldsType.4fb) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_54.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_54: type = splice_block %tuple.elem0.loc8_54.2 [symbolic = %tuple.elem0.loc8_54.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_53: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %tuple.elem0.loc8_54.2: type = tuple_access %T.ref.loc8_53, element0 [symbolic = %tuple.elem0.loc8_54.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%tuple.elem0.loc8_54.1 (%tuple.elem0) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %tuple.type) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.4fb
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
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_40.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.4fb)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %HoldsType.loc8_40.1 [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.ec9)]
// CHECK:STDOUT:   %x.param_patt.loc8_27.2: @F.%pattern_type.loc8_27 (%pattern_type.ec9) = value_param_pattern [symbolic = %x.param_patt.loc8_27.2 (constants.%x.param_patt.741)]
// CHECK:STDOUT:   %x.patt.loc8_27.2: @F.%pattern_type.loc8_27 (%pattern_type.ec9) = wrapper_binding_pattern x, %x.param_patt.loc8_27.2 [symbolic = %x.patt.loc8_27.2 (constants.%x.patt.991)]
// CHECK:STDOUT:   %tuple.elem0.loc8_54.1: type = tuple_access %T.loc8_7.1, element0 [symbolic = %tuple.elem0.loc8_54.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_51: type = pattern_type %tuple.elem0.loc8_54.1 [symbolic = %pattern_type.loc8_51 (constants.%pattern_type.251)]
// CHECK:STDOUT:   %a.param_patt.loc8_51.2: @F.%pattern_type.loc8_51 (%pattern_type.251) = value_param_pattern [symbolic = %a.param_patt.loc8_51.2 (constants.%a.param_patt.46f)]
// CHECK:STDOUT:   %a.patt.loc8_51.2: @F.%pattern_type.loc8_51 (%pattern_type.251) = wrapper_binding_pattern a, %a.param_patt.loc8_51.2 [symbolic = %a.patt.loc8_51.2 (constants.%a.patt.634)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27: <witness> = require_complete_type %HoldsType.loc8_40.1 [symbolic = %require_complete.loc8_27 (constants.%require_complete.4d4)]
// CHECK:STDOUT:   %require_complete.loc8_51: <witness> = require_complete_type %tuple.elem0.loc8_54.1 [symbolic = %require_complete.loc8_51 (constants.%require_complete.3a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_40.1 (%HoldsType.4fb), %a.param: @F.%tuple.elem0.loc8_54.1 (%tuple.elem0)) {
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
// CHECK:STDOUT:   %.loc13_25: %tuple.type = tuple_literal (%C.ref) [concrete = constants.%tuple.9e1]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C.ref) [concrete = constants.%tuple.9e1]
// CHECK:STDOUT:   %.loc13_26: %tuple.type = converted %.loc13_25, %tuple [concrete = constants.%tuple.9e1]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.9e1) [concrete = constants.%HoldsType.210]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.210 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.210 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.210 = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.9e1) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.210 = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.533]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.210 = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C to %.loc13_30.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_30.1, %.loc13_30.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.5: ref %C = temporary %.loc13_30.2, %.loc13_30.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_30.6: %C = acquire_value %.loc13_30.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_30.6)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call.loc13_30: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound.1a9(constants.%.115)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound.eeb(constants.%.533)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_30.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_30.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_8(%self.param: ref %HoldsType.210) {
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
// CHECK:STDOUT:   %HoldsType.loc8_40.1 => constants.%HoldsType.4fb
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.ec9
// CHECK:STDOUT:   %x.param_patt.loc8_27.2 => constants.%x.param_patt.741
// CHECK:STDOUT:   %x.patt.loc8_27.2 => constants.%x.patt.991
// CHECK:STDOUT:   %tuple.elem0.loc8_54.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_51 => constants.%pattern_type.251
// CHECK:STDOUT:   %a.param_patt.loc8_51.2 => constants.%a.param_patt.46f
// CHECK:STDOUT:   %a.patt.loc8_51.2 => constants.%a.patt.634
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.9e1) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%tuple.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.9e1) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%tuple.9e1
// CHECK:STDOUT:   %HoldsType.loc8_40.1 => constants.%HoldsType.210
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.2e7
// CHECK:STDOUT:   %x.param_patt.loc8_27.2 => constants.%x.param_patt.261
// CHECK:STDOUT:   %x.patt.loc8_27.2 => constants.%x.patt.c43
// CHECK:STDOUT:   %tuple.elem0.loc8_54.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_51 => constants.%pattern_type.98b
// CHECK:STDOUT:   %a.param_patt.loc8_51.2 => constants.%a.param_patt.c88
// CHECK:STDOUT:   %a.patt.loc8_51.2 => constants.%a.patt.4cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc8_51 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %pattern_type.136: type = pattern_type %struct_type.t [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %struct_type.t = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.323: type = class_type @HoldsType, @HoldsType(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.9c1: type = pattern_type %HoldsType.323 [symbolic]
// CHECK:STDOUT:   %x.param_patt.7e7: %pattern_type.9c1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.ebc: %pattern_type.9c1 = wrapper_binding_pattern x, %x.param_patt.7e7 [symbolic]
// CHECK:STDOUT:   %.841: type = struct_access %T, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4ae: type = pattern_type %.841 [symbolic]
// CHECK:STDOUT:   %a.param_patt.ad4: %pattern_type.4ae = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.68c: %pattern_type.4ae = wrapper_binding_pattern a, %a.param_patt.ad4 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.868: <witness> = require_complete_type %HoldsType.323 [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %.841 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.512: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.512 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.79a: type = pattern_type %HoldsType.512 [concrete]
// CHECK:STDOUT:   %x.param_patt.20b: %pattern_type.79a = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.d03: %pattern_type.79a = wrapper_binding_pattern x, %x.param_patt.20b [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.4cc: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %.3e2: ref %HoldsType.512 = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc13_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound.1a9: <bound method> = bound_method %.115, %Destroy.WithSelf.Op.403171.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound.191: <bound method> = bound_method %.3e2, %Destroy.WithSelf.Op.403171.3 [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_29: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_25: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_29.1: @F.%pattern_type.loc8_29 (%pattern_type.9c1) = value_param_pattern [symbolic = %x.param_patt.loc8_29.2 (constants.%x.param_patt.7e7)]
// CHECK:STDOUT:     %x.patt.loc8_29.1: @F.%pattern_type.loc8_29 (%pattern_type.9c1) = wrapper_binding_pattern x, %x.param_patt.loc8_29.1 [symbolic = %x.patt.loc8_29.2 (constants.%x.patt.ebc)]
// CHECK:STDOUT:     %a.param_patt.loc8_53.1: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = value_param_pattern [symbolic = %a.param_patt.loc8_53.2 (constants.%a.param_patt.ad4)]
// CHECK:STDOUT:     %a.patt.loc8_53.1: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = wrapper_binding_pattern a, %a.param_patt.loc8_53.1 [symbolic = %a.patt.loc8_53.2 (constants.%a.patt.68c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_14: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_42.1 (%HoldsType.323) = value_param call_param0
// CHECK:STDOUT:     %.loc8_42: type = splice_block %HoldsType.loc8_42.2 [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.323)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_41: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %HoldsType.loc8_42.2: type = class_type @HoldsType, @HoldsType(constants.%T) [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.323)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_42.1 (%HoldsType.323) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_56.1 (%.841) = value_param call_param1
// CHECK:STDOUT:     %.loc8_56.2: type = splice_block %.loc8_56.3 [symbolic = %.loc8_56.1 (constants.%.841)] {
// CHECK:STDOUT:       %T.ref.loc8_55: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_56.3: type = struct_access %T.ref.loc8_55, element0 [symbolic = %.loc8_56.1 (constants.%.841)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_56.1 (%.841) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %struct_type.t) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_18.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.323
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
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %HoldsType.loc8_42.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.323)]
// CHECK:STDOUT:   %pattern_type.loc8_29: type = pattern_type %HoldsType.loc8_42.1 [symbolic = %pattern_type.loc8_29 (constants.%pattern_type.9c1)]
// CHECK:STDOUT:   %x.param_patt.loc8_29.2: @F.%pattern_type.loc8_29 (%pattern_type.9c1) = value_param_pattern [symbolic = %x.param_patt.loc8_29.2 (constants.%x.param_patt.7e7)]
// CHECK:STDOUT:   %x.patt.loc8_29.2: @F.%pattern_type.loc8_29 (%pattern_type.9c1) = wrapper_binding_pattern x, %x.param_patt.loc8_29.2 [symbolic = %x.patt.loc8_29.2 (constants.%x.patt.ebc)]
// CHECK:STDOUT:   %.loc8_56.1: type = struct_access %T.loc8_7.1, element0 [symbolic = %.loc8_56.1 (constants.%.841)]
// CHECK:STDOUT:   %pattern_type.loc8_53: type = pattern_type %.loc8_56.1 [symbolic = %pattern_type.loc8_53 (constants.%pattern_type.4ae)]
// CHECK:STDOUT:   %a.param_patt.loc8_53.2: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = value_param_pattern [symbolic = %a.param_patt.loc8_53.2 (constants.%a.param_patt.ad4)]
// CHECK:STDOUT:   %a.patt.loc8_53.2: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = wrapper_binding_pattern a, %a.param_patt.loc8_53.2 [symbolic = %a.patt.loc8_53.2 (constants.%a.patt.68c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_29: <witness> = require_complete_type %HoldsType.loc8_42.1 [symbolic = %require_complete.loc8_29 (constants.%require_complete.868)]
// CHECK:STDOUT:   %require_complete.loc8_53: <witness> = require_complete_type %.loc8_56.1 [symbolic = %require_complete.loc8_53 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_42.1 (%HoldsType.323), %a.param: @F.%.loc8_56.1 (%.841)) {
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
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.512]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.512 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.512 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.512 = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.512 = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.3e2]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.512 = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C to %.loc13_33.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: init %C = converted %.loc13_33.1, %.loc13_33.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.5: ref %C = temporary %.loc13_33.2, %.loc13_33.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_33.6: %C = acquire_value %.loc13_33.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_33.6)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call.loc13_33: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound.1a9(constants.%.115)
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call.loc13_8: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound.191(constants.%.3e2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_33.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_33.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_8(%self.param: ref %HoldsType.512) {
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
// CHECK:STDOUT:   %HoldsType.loc8_42.1 => constants.%HoldsType.323
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.9c1
// CHECK:STDOUT:   %x.param_patt.loc8_29.2 => constants.%x.param_patt.7e7
// CHECK:STDOUT:   %x.patt.loc8_29.2 => constants.%x.patt.ebc
// CHECK:STDOUT:   %.loc8_56.1 => constants.%.841
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.4ae
// CHECK:STDOUT:   %a.param_patt.loc8_53.2 => constants.%a.param_patt.ad4
// CHECK:STDOUT:   %a.patt.loc8_53.2 => constants.%a.patt.68c
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
// CHECK:STDOUT:   %HoldsType.loc8_42.1 => constants.%HoldsType.512
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.79a
// CHECK:STDOUT:   %x.param_patt.loc8_29.2 => constants.%x.param_patt.20b
// CHECK:STDOUT:   %x.patt.loc8_29.2 => constants.%x.patt.d03
// CHECK:STDOUT:   %.loc8_56.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.98b
// CHECK:STDOUT:   %a.param_patt.loc8_53.2 => constants.%a.param_patt.c88
// CHECK:STDOUT:   %a.patt.loc8_53.2 => constants.%a.patt.4cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_29 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc8_53 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_class_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, type [concrete]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: type} [concrete]
// CHECK:STDOUT:   %complete_type.738: <witness> = complete_type_witness %struct_type.t [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
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
// CHECK:STDOUT:   %x.patt: %pattern_type.e53 = wrapper_binding_pattern x, %x.param_patt [symbolic]
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
// CHECK:STDOUT:   %Copy.impl_witness.f3e: <witness> = impl_witness imports.%Copy.impl_witness_table.aa5 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.f3e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.571: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.39e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.571, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %.0de: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%Class.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc30_13.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound: <bound method> = bound_method %.0de, %Destroy.WithSelf.Op.403171.3 [concrete]
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
// CHECK:STDOUT:   %Copy.impl_witness_table.aa5 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_18.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc8_18.1 (constants.%T.f25)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc21_7.1: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_7.2 (constants.%T.patt.125)]
// CHECK:STDOUT:     %x.param_patt.loc21_24.1: @F.%pattern_type (%pattern_type.e53) = value_param_pattern [symbolic = %x.param_patt.loc21_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc21_24.1: @F.%pattern_type (%pattern_type.e53) = wrapper_binding_pattern x, %x.param_patt.loc21_24.1 [symbolic = %x.patt.loc21_24.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = wrapper_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_9: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_7.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc21_37.1 (%HoldsType.3500da.1) = value_param call_param0
// CHECK:STDOUT:     %.loc21_37: type = splice_block %HoldsType.loc21_37.2 [symbolic = %HoldsType.loc21_37.1 (constants.%HoldsType.3500da.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc21_36: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %HoldsType.loc21_37.2: type = class_type @HoldsType, @HoldsType(constants.%T.f25) [symbolic = %HoldsType.loc21_37.1 (constants.%HoldsType.3500da.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc21_37.1 (%HoldsType.3500da.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc21_50: %Class = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:       %.loc21_51.2: ref type = class_element_access %T.ref.loc21_50, element0 [symbolic = %.loc21_51.1 (constants.%.9dd)]
// CHECK:STDOUT:       %.loc21_51.3: type = acquire_value %.loc21_51.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt.loc25_15.1: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc25_15.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc25_15.2: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_15.1 (constants.%c)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %field_decl: %Class.elem = field_decl t, element0, %.loc5 in [concrete] {
// CHECK:STDOUT:     %.loc5: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.t [concrete = constants.%complete_type.738]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .t = %field_decl
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
// CHECK:STDOUT:   %HoldsType.loc21_37.1: type = class_type @HoldsType, @HoldsType(%T.loc21_7.1) [symbolic = %HoldsType.loc21_37.1 (constants.%HoldsType.3500da.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc21_37.1 [symbolic = %pattern_type (constants.%pattern_type.e53)]
// CHECK:STDOUT:   %x.param_patt.loc21_24.2: @F.%pattern_type (%pattern_type.e53) = value_param_pattern [symbolic = %x.param_patt.loc21_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc21_24.2: @F.%pattern_type (%pattern_type.e53) = wrapper_binding_pattern x, %x.param_patt.loc21_24.2 [symbolic = %x.patt.loc21_24.2 (constants.%x.patt)]
// CHECK:STDOUT:   %.loc21_51.1: ref type = class_element_access %T.loc21_7.1, element0 [symbolic = %.loc21_51.1 (constants.%.9dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc21_37.1 [symbolic = %require_complete (constants.%require_complete.6f4d7d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc21_37.1 (%HoldsType.3500da.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%c.loc25_15.2: %Class) {
// CHECK:STDOUT:   %c.patt.loc25_15.2: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc25_15.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.loc25_15.1: %Class = symbolic_binding c, 0 [symbolic = %c.loc25_15.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc26_22.2: type = class_type @HoldsType, @HoldsType(%c.loc25_15.1) [symbolic = %HoldsType.loc26_22.2 (constants.%HoldsType.3500da.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc26_22.2 [symbolic = %require_complete (constants.%require_complete.6f4d7d.2)]
// CHECK:STDOUT:   %HoldsType.val: @G.%HoldsType.loc26_22.2 (%HoldsType.3500da.2) = struct_value () [symbolic = %HoldsType.val (constants.%HoldsType.val.7f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc26_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:     %c.ref: %Class = name_ref c, %c.loc25_15.2 [symbolic = %c.loc25_15.1 (constants.%c)]
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
// CHECK:STDOUT:   %impl.elem0: %.39e = impl_witness_access constants.%Copy.impl_witness.f3e, element0 [concrete = constants.%type.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound(constants.%.0de)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc30_13.1(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc30_13.2(%self.param: ref %struct_type.t) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc30_13.3(%self.param: ref %Class) {
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
// CHECK:STDOUT:   %HoldsType.loc21_37.1 => constants.%HoldsType.3500da.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e53
// CHECK:STDOUT:   %x.param_patt.loc21_24.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc21_24.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc21_51.1 => constants.%.9dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc25_15.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc25_15.1 => constants.%c
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
// CHECK:STDOUT:   %c.patt.loc25_15.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc25_15.1 => constants.%Class.val
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, type [concrete]
// CHECK:STDOUT:   %pattern_type.4fd: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %T.patt.c49: %pattern_type.4fd = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.225: %array_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = generic_class_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.b96: type = class_type @HoldsType, @HoldsType(%T.225) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.a68: type = pattern_type %HoldsType.b96 [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.a68 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.a68 = wrapper_binding_pattern x, %x.param_patt [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.00e: <witness> = require_complete_type %HoldsType.b96 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.9e1: %tuple.type.135 = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f3e: <witness> = impl_witness imports.%Copy.impl_witness_table.aa5 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.f3e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.571: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.39e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.571, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %.f9a: ref %array_type = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %HoldsType.d9e: type = class_type @HoldsType, @HoldsType(%array) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.d9e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc17_27.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.bound: <bound method> = bound_method %.f9a, %Destroy.WithSelf.Op.403171.2 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.aa5 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.4fd = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.c49)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_26: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc4_26 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.225)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_7.1: %pattern_type.4fd = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.c49)]
// CHECK:STDOUT:     %x.param_patt.loc12_33.1: @F.%pattern_type (%pattern_type.a68) = value_param_pattern [symbolic = %x.param_patt.loc12_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc12_33.1: @F.%pattern_type (%pattern_type.a68) = wrapper_binding_pattern x, %x.param_patt.loc12_33.1 [symbolic = %x.patt.loc12_33.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = wrapper_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_22: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc12_15: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %array_type: type = array_type %int_1, %.loc12_15 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.225)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc12_46.1 (%HoldsType.b96) = value_param call_param0
// CHECK:STDOUT:     %.loc12_46: type = splice_block %HoldsType.loc12_46.2 [symbolic = %HoldsType.loc12_46.1 (constants.%HoldsType.b96)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc12_45: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.225)]
// CHECK:STDOUT:       %HoldsType.loc12_46.2: type = class_type @HoldsType, @HoldsType(constants.%T.225) [symbolic = %HoldsType.loc12_46.1 (constants.%HoldsType.b96)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc12_46.1 (%HoldsType.b96) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc12_59: %array_type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.225)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.loc12_61.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_61.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_61.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_61.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_61.2: %i32 = converted %int_0, %.loc12_61.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:       %.loc12_62.1: ref %array_type = value_as_ref %T.ref.loc12_59
// CHECK:STDOUT:       %.loc12_62.2: ref type = array_index %.loc12_62.1, %.loc12_61.2
// CHECK:STDOUT:       %.loc12_62.3: type = acquire_value %.loc12_62.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %array_type) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.4fd = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.c49)]
// CHECK:STDOUT:   %T.loc4_18.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.225)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.b96
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
// CHECK:STDOUT:   %T.patt.loc12_7.2: %pattern_type.4fd = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.c49)]
// CHECK:STDOUT:   %T.loc12_7.1: %array_type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.225)]
// CHECK:STDOUT:   %HoldsType.loc12_46.1: type = class_type @HoldsType, @HoldsType(%T.loc12_7.1) [symbolic = %HoldsType.loc12_46.1 (constants.%HoldsType.b96)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc12_46.1 [symbolic = %pattern_type (constants.%pattern_type.a68)]
// CHECK:STDOUT:   %x.param_patt.loc12_33.2: @F.%pattern_type (%pattern_type.a68) = value_param_pattern [symbolic = %x.param_patt.loc12_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc12_33.2: @F.%pattern_type (%pattern_type.a68) = wrapper_binding_pattern x, %x.param_patt.loc12_33.2 [symbolic = %x.patt.loc12_33.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc12_46.1 [symbolic = %require_complete (constants.%require_complete.00e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc12_46.1 (%HoldsType.b96), %a.param: <error>) {
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
// CHECK:STDOUT:   %.loc17_25.1: %tuple.type.135 = tuple_literal (%C.ref) [concrete = constants.%tuple.9e1]
// CHECK:STDOUT:   %.loc17_36: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %.loc17_36 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0: %.39e = impl_witness_access constants.%Copy.impl_witness.f3e, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.2: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc17_25.3: ref type = array_index %.loc17_25.2, %int_0
// CHECK:STDOUT:   %.loc17_25.4: init type to %.loc17_25.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc17_25.5: init %array_type to %.loc17_25.2 = array_init (%.loc17_25.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.1: init %array_type = converted %.loc17_25.1, %.loc17_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_27.2: ref %array_type = temporary %.loc17_25.2, %.loc17_27.1 [concrete = constants.%.f9a]
// CHECK:STDOUT:   %.loc17_27.3: %array_type = acquire_value %.loc17_27.2 [concrete = constants.%array]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%array) [concrete = constants.%HoldsType.d9e]
// CHECK:STDOUT:   %.loc17_6.2: ref %HoldsType.d9e = temporary_storage
// CHECK:STDOUT:   %.loc17_6.3: init %HoldsType.d9e to %.loc17_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_8: init %HoldsType.d9e = converted %.loc17_6.1, %.loc17_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc17_48: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.Op.bound(constants.%.f9a)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc17_27.1(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc17_27.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.225) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.c49
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.225
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.225) {
// CHECK:STDOUT:   %T.patt.loc12_7.2 => constants.%T.patt.c49
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T.225
// CHECK:STDOUT:   %HoldsType.loc12_46.1 => constants.%HoldsType.b96
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a68
// CHECK:STDOUT:   %x.param_patt.loc12_33.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc12_33.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%array) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.c49
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%array
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
