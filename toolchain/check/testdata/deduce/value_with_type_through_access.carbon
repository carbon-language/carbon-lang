// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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
// CHECK:STDERR:        ╭─┤ fail_todo_class_access.carbon:[[@LINE+8]]:50
// CHECK:STDERR: ->     │ fn F[T: Class](unused x: HoldsType(T), unused a: T.t) {}
// CHECK:STDERR:        ·                                                  ┳━━
// CHECK:STDERR:        ·                                                  ┗━━┫ error: cannot evaluate
// CHECK:STDERR:        ·                                                     ┃        type expression
// CHECK:STDERR:        ·                                                     ┃        [TypeExprEvaluationFailure]
// CHECK:STDERR:        ╰────
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
// CHECK:STDERR:        ╭─┤ fail_todo_array_index.carbon:[[@LINE+10]]:59
// CHECK:STDERR: ->     │ fn F[T: array(type, 1)](unused x: HoldsType(T), unused a: T[0]) {}
// CHECK:STDERR:        ·                                                           ┳━━━
// CHECK:STDERR:        ·                                                           ┗━━┫ error: cannot
// CHECK:STDERR:        ·                                                              ┃        evaluate
// CHECK:STDERR:        ·                                                              ┃        type
// CHECK:STDERR:        ·                                                              ┃        expression
// CHECK:STDERR:        ·                                                              ┃        [TypeExprEvaluationFailure]
// CHECK:STDERR:        ╰────
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
// CHECK:STDOUT:   %T.patt.d26: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.198: %tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = parameterized_class_name_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.7c6: type = class_type @HoldsType, @HoldsType(%T.198) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %HoldsType.7c6 [symbolic]
// CHECK:STDOUT:   %x.param_patt.c68: %pattern_type.831 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.efd: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt.c68 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.elem0: type = tuple_access %T.198, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.251: type = pattern_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %a.param_patt.46f: %pattern_type.251 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.634: %pattern_type.251 = wrapper_binding_pattern a, %a.param_patt.46f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1ff: <witness> = require_complete_type %HoldsType.7c6 [symbolic]
// CHECK:STDOUT:   %require_complete.3a8: <witness> = require_complete_type %tuple.elem0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.9e1: %tuple.type = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.f9f: type = class_type @HoldsType, @HoldsType(%tuple.9e1) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.f9f = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6d1: type = pattern_type %HoldsType.f9f [concrete]
// CHECK:STDOUT:   %x.param_patt.95c: %pattern_type.6d1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.b37: %pattern_type.6d1 = wrapper_binding_pattern x, %x.param_patt.95c [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.4cc: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.9e1) [concrete]
// CHECK:STDOUT:   %.f6a: ref %HoldsType.f9f = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_30.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb2: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8ca: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.eb2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.61c: %SubobjectDestroy.type = facet_value %HoldsType.f9f, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.5a7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.61c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.642: %Destroy.type = facet_value %HoldsType.f9f, (%DestroyImpl.impl_witness.5a7, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.205: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.642) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_21: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc4_27.2: %tuple.type = tuple_literal (%.loc4_21) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc4_27.3: type = converted %.loc4_27.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.198)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:     %x.param_patt.loc8_27.1: @F.%pattern_type.loc8_27 (%pattern_type.831) = value_param_pattern [symbolic = %x.param_patt.loc8_27.2 (constants.%x.param_patt.c68)]
// CHECK:STDOUT:     %x.patt.loc8_27.1: @F.%pattern_type.loc8_27 (%pattern_type.831) = wrapper_binding_pattern x, %x.param_patt.loc8_27.1 [symbolic = %x.patt.loc8_27.2 (constants.%x.patt.efd)]
// CHECK:STDOUT:     %a.param_patt.loc8_51.1: @F.%pattern_type.loc8_51 (%pattern_type.251) = value_param_pattern [symbolic = %a.param_patt.loc8_51.2 (constants.%a.param_patt.46f)]
// CHECK:STDOUT:     %a.patt.loc8_51.1: @F.%pattern_type.loc8_51 (%pattern_type.251) = wrapper_binding_pattern a, %a.param_patt.loc8_51.1 [symbolic = %a.patt.loc8_51.2 (constants.%a.patt.634)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_16.1: type = splice_block %.loc8_16.3 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_10: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.loc8_16.2: %tuple.type = tuple_literal (%.loc8_10) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:       %.loc8_16.3: type = converted %.loc8_16.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.198)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_40.1 (%HoldsType.7c6) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40: type = splice_block %HoldsType.loc8_40.2 [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.7c6)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_39: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.198)]
// CHECK:STDOUT:       %HoldsType.loc8_40.2: type = class_type @HoldsType, @HoldsType(constants.%T.198) [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.7c6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_40.1 (%HoldsType.7c6) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%tuple.elem0.loc8_54.1 (%tuple.elem0) = value_param call_param1
// CHECK:STDOUT:     %.loc8_54: type = splice_block %tuple.elem0.loc8_54.2 [symbolic = %tuple.elem0.loc8_54.1 (constants.%tuple.elem0)] {
// CHECK:STDOUT:       %T.ref.loc8_53: %tuple.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.198)]
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
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   %T.loc4_18.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.198)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.7c6
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
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.ddf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.d26)]
// CHECK:STDOUT:   %T.loc8_7.1: %tuple.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.198)]
// CHECK:STDOUT:   %HoldsType.loc8_40.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_40.1 (constants.%HoldsType.7c6)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %HoldsType.loc8_40.1 [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.831)]
// CHECK:STDOUT:   %x.param_patt.loc8_27.2: @F.%pattern_type.loc8_27 (%pattern_type.831) = value_param_pattern [symbolic = %x.param_patt.loc8_27.2 (constants.%x.param_patt.c68)]
// CHECK:STDOUT:   %x.patt.loc8_27.2: @F.%pattern_type.loc8_27 (%pattern_type.831) = wrapper_binding_pattern x, %x.param_patt.loc8_27.2 [symbolic = %x.patt.loc8_27.2 (constants.%x.patt.efd)]
// CHECK:STDOUT:   %tuple.elem0.loc8_54.1: type = tuple_access %T.loc8_7.1, element0 [symbolic = %tuple.elem0.loc8_54.1 (constants.%tuple.elem0)]
// CHECK:STDOUT:   %pattern_type.loc8_51: type = pattern_type %tuple.elem0.loc8_54.1 [symbolic = %pattern_type.loc8_51 (constants.%pattern_type.251)]
// CHECK:STDOUT:   %a.param_patt.loc8_51.2: @F.%pattern_type.loc8_51 (%pattern_type.251) = value_param_pattern [symbolic = %a.param_patt.loc8_51.2 (constants.%a.param_patt.46f)]
// CHECK:STDOUT:   %a.patt.loc8_51.2: @F.%pattern_type.loc8_51 (%pattern_type.251) = wrapper_binding_pattern a, %a.param_patt.loc8_51.2 [symbolic = %a.patt.loc8_51.2 (constants.%a.patt.634)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27: <witness> = require_complete_type %HoldsType.loc8_40.1 [symbolic = %require_complete.loc8_27 (constants.%require_complete.1ff)]
// CHECK:STDOUT:   %require_complete.loc8_51: <witness> = require_complete_type %tuple.elem0.loc8_54.1 [symbolic = %require_complete.loc8_51 (constants.%require_complete.3a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_40.1 (%HoldsType.7c6), %a.param: @F.%tuple.elem0.loc8_54.1 (%tuple.elem0)) {
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
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%tuple.9e1) [concrete = constants.%HoldsType.f9f]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.f9f = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.f9f to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.f9f = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.9e1) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.f9f = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.f6a]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.f9f = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_30.3: init %C to %.loc13_30.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_30.1, %.loc13_30.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_30.5: ref %C = temporary %.loc13_30.2, %.loc13_30.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_30.6: %C = acquire_value %.loc13_30.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_30.6)
// CHECK:STDOUT:   %Destroy.facet.loc13_30.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %.loc13_30.7: %Destroy.type = converted constants.%C, %Destroy.facet.loc13_30.1 [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %Destroy.facet.loc13_30.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %.loc13_30.8: %Destroy.type = converted constants.%C, %Destroy.facet.loc13_30.2 [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.eb2) [concrete = constants.%SelfDestruct.specific_fn.8ca]
// CHECK:STDOUT:   %SelfDestruct.call.loc13_30: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_30.5)
// CHECK:STDOUT:   %Destroy.facet.loc13_8.1: %Destroy.type = facet_value constants.%HoldsType.f9f, (constants.%DestroyImpl.impl_witness.5a7, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.642]
// CHECK:STDOUT:   %.loc13_8.4: %Destroy.type = converted constants.%HoldsType.f9f, %Destroy.facet.loc13_8.1 [concrete = constants.%Destroy.facet.642]
// CHECK:STDOUT:   %Destroy.facet.loc13_8.2: %Destroy.type = facet_value constants.%HoldsType.f9f, (constants.%DestroyImpl.impl_witness.5a7, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.642]
// CHECK:STDOUT:   %.loc13_8.5: %Destroy.type = converted constants.%HoldsType.f9f, %Destroy.facet.loc13_8.2 [concrete = constants.%Destroy.facet.642]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.642) [concrete = constants.%SelfDestruct.specific_fn.205]
// CHECK:STDOUT:   %SelfDestruct.call.loc13_8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_30.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_30.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_8(%self.param: ref %HoldsType.f9f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.198) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.198
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.198) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T.198
// CHECK:STDOUT:   %HoldsType.loc8_40.1 => constants.%HoldsType.7c6
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.831
// CHECK:STDOUT:   %x.param_patt.loc8_27.2 => constants.%x.param_patt.c68
// CHECK:STDOUT:   %x.patt.loc8_27.2 => constants.%x.patt.efd
// CHECK:STDOUT:   %tuple.elem0.loc8_54.1 => constants.%tuple.elem0
// CHECK:STDOUT:   %pattern_type.loc8_51 => constants.%pattern_type.251
// CHECK:STDOUT:   %a.param_patt.loc8_51.2 => constants.%a.param_patt.46f
// CHECK:STDOUT:   %a.patt.loc8_51.2 => constants.%a.patt.634
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%tuple.9e1) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%tuple.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.9e1) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.d26
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%tuple.9e1
// CHECK:STDOUT:   %HoldsType.loc8_40.1 => constants.%HoldsType.f9f
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.6d1
// CHECK:STDOUT:   %x.param_patt.loc8_27.2 => constants.%x.param_patt.95c
// CHECK:STDOUT:   %x.patt.loc8_27.2 => constants.%x.patt.b37
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
// CHECK:STDOUT:   %T.patt.b23: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.6cb: %struct_type.t = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = parameterized_class_name_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.73d: type = class_type @HoldsType, @HoldsType(%T.6cb) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.92c: type = pattern_type %HoldsType.73d [symbolic]
// CHECK:STDOUT:   %x.param_patt.601: %pattern_type.92c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a07: %pattern_type.92c = wrapper_binding_pattern x, %x.param_patt.601 [symbolic]
// CHECK:STDOUT:   %.841: type = struct_access %T.6cb, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4ae: type = pattern_type %.841 [symbolic]
// CHECK:STDOUT:   %a.param_patt.ad4: %pattern_type.4ae = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.68c: %pattern_type.4ae = wrapper_binding_pattern a, %a.param_patt.ad4 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e33: <witness> = require_complete_type %HoldsType.73d [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %.841 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %HoldsType.0a4: type = class_type @HoldsType, @HoldsType(%struct) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c28: type = pattern_type %HoldsType.0a4 [concrete]
// CHECK:STDOUT:   %x.param_patt.0bb: %pattern_type.c28 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.b56: %pattern_type.c28 = wrapper_binding_pattern x, %x.param_patt.0bb [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt.c88: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.4cc: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt.c88 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%struct) [concrete]
// CHECK:STDOUT:   %.7bf: ref %HoldsType.0a4 = temporary invalid, %HoldsType.val [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_33.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb2: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8ca: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.eb2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.016: %SubobjectDestroy.type = facet_value %HoldsType.0a4, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.675: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.016) [concrete]
// CHECK:STDOUT:   %Destroy.facet.39a: %Destroy.type = facet_value %HoldsType.0a4, (%DestroyImpl.impl_witness.675, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8d9: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.39a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.b23)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_29: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_25: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.6cb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.b23)]
// CHECK:STDOUT:     %x.param_patt.loc8_29.1: @F.%pattern_type.loc8_29 (%pattern_type.92c) = value_param_pattern [symbolic = %x.param_patt.loc8_29.2 (constants.%x.param_patt.601)]
// CHECK:STDOUT:     %x.patt.loc8_29.1: @F.%pattern_type.loc8_29 (%pattern_type.92c) = wrapper_binding_pattern x, %x.param_patt.loc8_29.1 [symbolic = %x.patt.loc8_29.2 (constants.%x.patt.a07)]
// CHECK:STDOUT:     %a.param_patt.loc8_53.1: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = value_param_pattern [symbolic = %a.param_patt.loc8_53.2 (constants.%a.param_patt.ad4)]
// CHECK:STDOUT:     %a.patt.loc8_53.1: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = wrapper_binding_pattern a, %a.param_patt.loc8_53.1 [symbolic = %a.patt.loc8_53.2 (constants.%a.patt.68c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18: type = splice_block %struct_type.t [concrete = constants.%struct_type.t] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_14: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %struct_type.t: type = struct_type {.t: type} [concrete = constants.%struct_type.t]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.6cb)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc8_42.1 (%HoldsType.73d) = value_param call_param0
// CHECK:STDOUT:     %.loc8_42: type = splice_block %HoldsType.loc8_42.2 [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.73d)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc8_41: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.6cb)]
// CHECK:STDOUT:       %HoldsType.loc8_42.2: type = class_type @HoldsType, @HoldsType(constants.%T.6cb) [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.73d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc8_42.1 (%HoldsType.73d) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: @F.%.loc8_56.1 (%.841) = value_param call_param1
// CHECK:STDOUT:     %.loc8_56.2: type = splice_block %.loc8_56.3 [symbolic = %.loc8_56.1 (constants.%.841)] {
// CHECK:STDOUT:       %T.ref.loc8_55: %struct_type.t = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.6cb)]
// CHECK:STDOUT:       %.loc8_56.3: type = struct_access %T.ref.loc8_55, element0 [symbolic = %.loc8_56.1 (constants.%.841)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc8_56.1 (%.841) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %struct_type.t) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.b23)]
// CHECK:STDOUT:   %T.loc4_18.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.6cb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.73d
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
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.136 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.b23)]
// CHECK:STDOUT:   %T.loc8_7.1: %struct_type.t = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.6cb)]
// CHECK:STDOUT:   %HoldsType.loc8_42.1: type = class_type @HoldsType, @HoldsType(%T.loc8_7.1) [symbolic = %HoldsType.loc8_42.1 (constants.%HoldsType.73d)]
// CHECK:STDOUT:   %pattern_type.loc8_29: type = pattern_type %HoldsType.loc8_42.1 [symbolic = %pattern_type.loc8_29 (constants.%pattern_type.92c)]
// CHECK:STDOUT:   %x.param_patt.loc8_29.2: @F.%pattern_type.loc8_29 (%pattern_type.92c) = value_param_pattern [symbolic = %x.param_patt.loc8_29.2 (constants.%x.param_patt.601)]
// CHECK:STDOUT:   %x.patt.loc8_29.2: @F.%pattern_type.loc8_29 (%pattern_type.92c) = wrapper_binding_pattern x, %x.param_patt.loc8_29.2 [symbolic = %x.patt.loc8_29.2 (constants.%x.patt.a07)]
// CHECK:STDOUT:   %.loc8_56.1: type = struct_access %T.loc8_7.1, element0 [symbolic = %.loc8_56.1 (constants.%.841)]
// CHECK:STDOUT:   %pattern_type.loc8_53: type = pattern_type %.loc8_56.1 [symbolic = %pattern_type.loc8_53 (constants.%pattern_type.4ae)]
// CHECK:STDOUT:   %a.param_patt.loc8_53.2: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = value_param_pattern [symbolic = %a.param_patt.loc8_53.2 (constants.%a.param_patt.ad4)]
// CHECK:STDOUT:   %a.patt.loc8_53.2: @F.%pattern_type.loc8_53 (%pattern_type.4ae) = wrapper_binding_pattern a, %a.param_patt.loc8_53.2 [symbolic = %a.patt.loc8_53.2 (constants.%a.patt.68c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_29: <witness> = require_complete_type %HoldsType.loc8_42.1 [symbolic = %require_complete.loc8_29 (constants.%require_complete.e33)]
// CHECK:STDOUT:   %require_complete.loc8_53: <witness> = require_complete_type %.loc8_56.1 [symbolic = %require_complete.loc8_53 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc8_42.1 (%HoldsType.73d), %a.param: @F.%.loc8_56.1 (%.841)) {
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
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%struct) [concrete = constants.%HoldsType.0a4]
// CHECK:STDOUT:   %.loc13_6.2: ref %HoldsType.0a4 = temporary_storage
// CHECK:STDOUT:   %.loc13_6.3: init %HoldsType.0a4 to %.loc13_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_8.1: init %HoldsType.0a4 = converted %.loc13_6.1, %.loc13_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_8.2: ref %HoldsType.0a4 = temporary %.loc13_6.2, %.loc13_8.1 [concrete = constants.%.7bf]
// CHECK:STDOUT:   %.loc13_8.3: %HoldsType.0a4 = acquire_value %.loc13_8.2 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc13_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_33.3: init %C to %.loc13_33.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.4: init %C = converted %.loc13_33.1, %.loc13_33.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_33.5: ref %C = temporary %.loc13_33.2, %.loc13_33.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_33.6: %C = acquire_value %.loc13_33.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc13_8.3, %.loc13_33.6)
// CHECK:STDOUT:   %Destroy.facet.loc13_33.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %.loc13_33.7: %Destroy.type = converted constants.%C, %Destroy.facet.loc13_33.1 [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %Destroy.facet.loc13_33.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %.loc13_33.8: %Destroy.type = converted constants.%C, %Destroy.facet.loc13_33.2 [concrete = constants.%Destroy.facet.eb2]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.eb2) [concrete = constants.%SelfDestruct.specific_fn.8ca]
// CHECK:STDOUT:   %SelfDestruct.call.loc13_33: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_33.5)
// CHECK:STDOUT:   %Destroy.facet.loc13_8.1: %Destroy.type = facet_value constants.%HoldsType.0a4, (constants.%DestroyImpl.impl_witness.675, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.39a]
// CHECK:STDOUT:   %.loc13_8.4: %Destroy.type = converted constants.%HoldsType.0a4, %Destroy.facet.loc13_8.1 [concrete = constants.%Destroy.facet.39a]
// CHECK:STDOUT:   %Destroy.facet.loc13_8.2: %Destroy.type = facet_value constants.%HoldsType.0a4, (constants.%DestroyImpl.impl_witness.675, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.39a]
// CHECK:STDOUT:   %.loc13_8.5: %Destroy.type = converted constants.%HoldsType.0a4, %Destroy.facet.loc13_8.2 [concrete = constants.%Destroy.facet.39a]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.39a) [concrete = constants.%SelfDestruct.specific_fn.8d9]
// CHECK:STDOUT:   %SelfDestruct.call.loc13_8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_33.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_33.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_8(%self.param: ref %HoldsType.0a4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.6cb) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.b23
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.6cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.6cb) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.b23
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T.6cb
// CHECK:STDOUT:   %HoldsType.loc8_42.1 => constants.%HoldsType.73d
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.92c
// CHECK:STDOUT:   %x.param_patt.loc8_29.2 => constants.%x.param_patt.601
// CHECK:STDOUT:   %x.patt.loc8_29.2 => constants.%x.patt.a07
// CHECK:STDOUT:   %.loc8_56.1 => constants.%.841
// CHECK:STDOUT:   %pattern_type.loc8_53 => constants.%pattern_type.4ae
// CHECK:STDOUT:   %a.param_patt.loc8_53.2 => constants.%a.param_patt.ad4
// CHECK:STDOUT:   %a.patt.loc8_53.2 => constants.%a.patt.68c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%struct) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.b23
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.b23
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%struct
// CHECK:STDOUT:   %HoldsType.loc8_42.1 => constants.%HoldsType.0a4
// CHECK:STDOUT:   %pattern_type.loc8_29 => constants.%pattern_type.c28
// CHECK:STDOUT:   %x.param_patt.loc8_29.2 => constants.%x.param_patt.0bb
// CHECK:STDOUT:   %x.patt.loc8_29.2 => constants.%x.patt.b56
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
// CHECK:STDOUT:   %HoldsType.type: type = parameterized_class_name_type @HoldsType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.ef239b.1: type = class_type @HoldsType, @HoldsType(%T.f25) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.d63: type = pattern_type %HoldsType.ef239b.1 [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d63 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.d63 = wrapper_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %.9dd: ref type = class_element_access %T.f25, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.8b59ba.1: <witness> = require_complete_type %HoldsType.ef239b.1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic]
// CHECK:STDOUT:   %c: %Class = symbolic_binding c, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.ef239b.2: type = class_type @HoldsType, @HoldsType(%c) [symbolic]
// CHECK:STDOUT:   %require_complete.8b59ba.2: <witness> = require_complete_type %HoldsType.ef239b.2 [symbolic]
// CHECK:STDOUT:   %HoldsType.val.16e: %HoldsType.ef239b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.t = struct_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f3e: <witness> = impl_witness imports.%Copy.impl_witness_table.aa5 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.f3e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ca9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cd2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ca9, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%C) [concrete]
// CHECK:STDOUT:   %.0de: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%Class.val) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_13.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.029: %Destroy.type = facet_value type, (%DestroyImpl.impl_witness.a54, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.fe0: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.029) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_13.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3a1: %SubobjectDestroy.type = facet_value %Class, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.66c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3a1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.728: %Destroy.type = facet_value %Class, (%DestroyImpl.impl_witness.66c, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.744: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.728) [concrete]
// CHECK:STDOUT:   %HoldsType.ddc: type = class_type @HoldsType, @HoldsType(%Class.val) [concrete]
// CHECK:STDOUT:   %HoldsType.val.046: %HoldsType.ddc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.aa5 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc25_7.1: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_7.2 (constants.%T.patt.125)]
// CHECK:STDOUT:     %x.param_patt.loc25_24.1: @F.%pattern_type (%pattern_type.d63) = value_param_pattern [symbolic = %x.param_patt.loc25_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc25_24.1: @F.%pattern_type (%pattern_type.d63) = wrapper_binding_pattern x, %x.param_patt.loc25_24.1 [symbolic = %x.patt.loc25_24.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = wrapper_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_9: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_7.2: %Class = symbolic_binding T, 0 [symbolic = %T.loc25_7.1 (constants.%T.f25)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc25_37.1 (%HoldsType.ef239b.1) = value_param call_param0
// CHECK:STDOUT:     %.loc25_37: type = splice_block %HoldsType.loc25_37.2 [symbolic = %HoldsType.loc25_37.1 (constants.%HoldsType.ef239b.1)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc25_36: %Class = name_ref T, %T.loc25_7.2 [symbolic = %T.loc25_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %HoldsType.loc25_37.2: type = class_type @HoldsType, @HoldsType(constants.%T.f25) [symbolic = %HoldsType.loc25_37.1 (constants.%HoldsType.ef239b.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc25_37.1 (%HoldsType.ef239b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc25_50: %Class = name_ref T, %T.loc25_7.2 [symbolic = %T.loc25_7.1 (constants.%T.f25)]
// CHECK:STDOUT:       %t.ref: %Class.elem = name_ref t, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:       %.loc25_51.2: ref type = class_element_access %T.ref.loc25_50, element0 [symbolic = %.loc25_51.1 (constants.%.9dd)]
// CHECK:STDOUT:       %.loc25_51.3: type = acquire_value %.loc25_51.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt.loc29_15.1: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc29_15.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29: type = splice_block %Class.ref [concrete = constants.%Class] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc29_15.2: %Class = symbolic_binding c, 0 [symbolic = %c.loc29_15.1 (constants.%c)]
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
// CHECK:STDOUT:     .Self = constants.%HoldsType.ef239b.1
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
// CHECK:STDOUT: generic fn @F(%T.loc25_7.2: %Class) {
// CHECK:STDOUT:   %T.patt.loc25_7.2: %pattern_type.f15 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_7.2 (constants.%T.patt.125)]
// CHECK:STDOUT:   %T.loc25_7.1: %Class = symbolic_binding T, 0 [symbolic = %T.loc25_7.1 (constants.%T.f25)]
// CHECK:STDOUT:   %HoldsType.loc25_37.1: type = class_type @HoldsType, @HoldsType(%T.loc25_7.1) [symbolic = %HoldsType.loc25_37.1 (constants.%HoldsType.ef239b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc25_37.1 [symbolic = %pattern_type (constants.%pattern_type.d63)]
// CHECK:STDOUT:   %x.param_patt.loc25_24.2: @F.%pattern_type (%pattern_type.d63) = value_param_pattern [symbolic = %x.param_patt.loc25_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc25_24.2: @F.%pattern_type (%pattern_type.d63) = wrapper_binding_pattern x, %x.param_patt.loc25_24.2 [symbolic = %x.patt.loc25_24.2 (constants.%x.patt)]
// CHECK:STDOUT:   %.loc25_51.1: ref type = class_element_access %T.loc25_7.1, element0 [symbolic = %.loc25_51.1 (constants.%.9dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc25_37.1 [symbolic = %require_complete (constants.%require_complete.8b59ba.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc25_37.1 (%HoldsType.ef239b.1), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%c.loc29_15.2: %Class) {
// CHECK:STDOUT:   %c.patt.loc29_15.2: %pattern_type.f15 = symbolic_binding_pattern c, 0 [symbolic = %c.patt.loc29_15.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.loc29_15.1: %Class = symbolic_binding c, 0 [symbolic = %c.loc29_15.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc30_22.2: type = class_type @HoldsType, @HoldsType(%c.loc29_15.1) [symbolic = %HoldsType.loc30_22.2 (constants.%HoldsType.ef239b.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc30_22.2 [symbolic = %require_complete (constants.%require_complete.8b59ba.2)]
// CHECK:STDOUT:   %HoldsType.val: @G.%HoldsType.loc30_22.2 (%HoldsType.ef239b.2) = struct_value () [symbolic = %HoldsType.val (constants.%HoldsType.val.16e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc30_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:     %c.ref: %Class = name_ref c, %c.loc29_15.2 [symbolic = %c.loc29_15.1 (constants.%c)]
// CHECK:STDOUT:     %HoldsType.loc30_22.1: type = class_type @HoldsType, @HoldsType(constants.%c) [symbolic = %HoldsType.loc30_22.2 (constants.%HoldsType.ef239b.2)]
// CHECK:STDOUT:     %.loc30_6.2: ref @G.%HoldsType.loc30_22.2 (%HoldsType.ef239b.2) = temporary_storage
// CHECK:STDOUT:     %.loc30_6.3: init @G.%HoldsType.loc30_22.2 (%HoldsType.ef239b.2) to %.loc30_6.2 = class_init () [symbolic = %HoldsType.val (constants.%HoldsType.val.16e)]
// CHECK:STDOUT:     %.loc30_8: init @G.%HoldsType.loc30_22.2 (%HoldsType.ef239b.2) = converted %.loc30_6.1, %.loc30_6.3 [symbolic = %HoldsType.val (constants.%HoldsType.val.16e)]
// CHECK:STDOUT:     %.loc30_26: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc34_12.1: %struct_type.t = struct_literal (%C.ref) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.cd2 = impl_witness_access constants.%Copy.impl_witness.f3e, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc34_12.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc34_12.3: ref type = class_element_access %.loc34_12.2, element0
// CHECK:STDOUT:   %.loc34_12.4: init type to %.loc34_12.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc34_12.5: init %Class to %.loc34_12.2 = class_init (%.loc34_12.4) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc34_13.1: init %Class = converted %.loc34_12.1, %.loc34_12.5 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc34_13.2: ref %Class = temporary %.loc34_12.2, %.loc34_13.1 [concrete = constants.%.0de]
// CHECK:STDOUT:   %.loc34_13.3: %Class = acquire_value %.loc34_13.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%Class.val) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   %Destroy.facet.loc34_13.1: %Destroy.type = facet_value constants.%Class, (constants.%DestroyImpl.impl_witness.66c, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.728]
// CHECK:STDOUT:   %.loc34_13.4: %Destroy.type = converted constants.%Class, %Destroy.facet.loc34_13.1 [concrete = constants.%Destroy.facet.728]
// CHECK:STDOUT:   %Destroy.facet.loc34_13.2: %Destroy.type = facet_value constants.%Class, (constants.%DestroyImpl.impl_witness.66c, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.728]
// CHECK:STDOUT:   %.loc34_13.5: %Destroy.type = converted constants.%Class, %Destroy.facet.loc34_13.2 [concrete = constants.%Destroy.facet.728]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.728) [concrete = constants.%SelfDestruct.specific_fn.744]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_13.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_13.1(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_13.2(%self.param: ref %struct_type.t) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_13.1: ref type = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc34_13.1: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc34_13.2: %Destroy.type = converted type, %Destroy.facet.loc34_13.1 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %Destroy.facet.loc34_13.2: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc34_13.3: %Destroy.type = converted type, %Destroy.facet.loc34_13.2 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.029) [concrete = constants.%SelfDestruct.specific_fn.fe0]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_13.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_13.3(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %Class.elem = name_ref t, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc34_13.1: ref type = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc34_13.1: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc34_13.2: %Destroy.type = converted type, %Destroy.facet.loc34_13.1 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %Destroy.facet.loc34_13.2: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc34_13.3: %Destroy.type = converted type, %Destroy.facet.loc34_13.2 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.029) [concrete = constants.%SelfDestruct.specific_fn.fe0]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc34_13.1)
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
// CHECK:STDOUT:   %T.patt.loc25_7.2 => constants.%T.patt.125
// CHECK:STDOUT:   %T.loc25_7.1 => constants.%T.f25
// CHECK:STDOUT:   %HoldsType.loc25_37.1 => constants.%HoldsType.ef239b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d63
// CHECK:STDOUT:   %x.param_patt.loc25_24.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc25_24.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc25_51.1 => constants.%.9dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc29_15.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc29_15.1 => constants.%c
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
// CHECK:STDOUT:   %c.patt.loc29_15.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc29_15.1 => constants.%Class.val
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HoldsType.loc30_22.2 => constants.%HoldsType.ddc
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %HoldsType.val => constants.%HoldsType.val.046
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
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.612: type = class_type @Array, @Array(type, %int_1) [concrete]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %Array.612 [concrete]
// CHECK:STDOUT:   %T.patt.bb1: %pattern_type.809 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.6be: %Array.612 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HoldsType.type: type = parameterized_class_name_type @HoldsType [concrete]
// CHECK:STDOUT:   %HoldsType.generic: %HoldsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %HoldsType.9dd: type = class_type @HoldsType, @HoldsType(%T.6be) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.dc7: type = pattern_type %HoldsType.9dd [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.dc7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.dc7 = wrapper_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.501: <witness> = require_complete_type %HoldsType.9dd [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.9e1: %tuple.type.135 = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f3e: <witness> = impl_witness imports.%Copy.impl_witness_table.aa5 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.f3e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ca9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cd2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ca9, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %C, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %array: %Array.612 = tuple_value (%C) [concrete]
// CHECK:STDOUT:   %.345: ref %Array.612 = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %HoldsType.00d: type = class_type @HoldsType, @HoldsType(%array) [concrete]
// CHECK:STDOUT:   %HoldsType.val: %HoldsType.00d = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.edc: %SubobjectDestroy.type = facet_value %Array.612, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e45: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.edc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.612, (%DestroyImpl.impl_witness.e45, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.aa5 = impl_witness_table (%Core.import_ref.9c3), @type.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_18.1: %pattern_type.809 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.bb1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33: type = splice_block %Array [concrete = constants.%Array.612] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_26: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %Array: type = class_type @Array, @Array(type, constants.%int_1) [concrete = constants.%Array.612]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: %Array.612 = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.6be)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc18_7.1: %pattern_type.809 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_7.2 (constants.%T.patt.bb1)]
// CHECK:STDOUT:     %x.param_patt.loc18_33.1: @F.%pattern_type (%pattern_type.dc7) = value_param_pattern [symbolic = %x.param_patt.loc18_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc18_33.1: @F.%pattern_type (%pattern_type.dc7) = wrapper_binding_pattern x, %x.param_patt.loc18_33.1 [symbolic = %x.patt.loc18_33.2 (constants.%x.patt)]
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %a.patt: <error> = wrapper_binding_pattern a, %a.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_22: type = splice_block %Array [concrete = constants.%Array.612] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc18_15: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %Array: type = class_type @Array, @Array(type, constants.%int_1) [concrete = constants.%Array.612]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_7.2: %Array.612 = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.6be)]
// CHECK:STDOUT:     %x.param: @F.%HoldsType.loc18_46.1 (%HoldsType.9dd) = value_param call_param0
// CHECK:STDOUT:     %.loc18_46: type = splice_block %HoldsType.loc18_46.2 [symbolic = %HoldsType.loc18_46.1 (constants.%HoldsType.9dd)] {
// CHECK:STDOUT:       %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:       %T.ref.loc18_45: %Array.612 = name_ref T, %T.loc18_7.2 [symbolic = %T.loc18_7.1 (constants.%T.6be)]
// CHECK:STDOUT:       %HoldsType.loc18_46.2: type = class_type @HoldsType, @HoldsType(constants.%T.6be) [symbolic = %HoldsType.loc18_46.1 (constants.%HoldsType.9dd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%HoldsType.loc18_46.1 (%HoldsType.9dd) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %a.param: <error> = value_param call_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %T.ref.loc18_59: %Array.612 = name_ref T, %T.loc18_7.2 [symbolic = %T.loc18_7.1 (constants.%T.6be)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:       %bound_method.loc18_61.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc18_61.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc18_61.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %.loc18_61.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %.loc18_61.2: %i32 = converted %int_0, %.loc18_61.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %.loc18_62.1: ref %Array.612 = value_as_ref %T.ref.loc18_59
// CHECK:STDOUT:       %.loc18_62.2: ref type = array_index %.loc18_62.1, %.loc18_61.2
// CHECK:STDOUT:       %.loc18_62.3: type = acquire_value %.loc18_62.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @HoldsType(%T.loc4_18.2: %Array.612) {
// CHECK:STDOUT:   %T.patt.loc4_18.2: %pattern_type.809 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_18.2 (constants.%T.patt.bb1)]
// CHECK:STDOUT:   %T.loc4_18.1: %Array.612 = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T.6be)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%HoldsType.9dd
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
// CHECK:STDOUT: generic fn @F(%T.loc18_7.2: %Array.612) {
// CHECK:STDOUT:   %T.patt.loc18_7.2: %pattern_type.809 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_7.2 (constants.%T.patt.bb1)]
// CHECK:STDOUT:   %T.loc18_7.1: %Array.612 = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.6be)]
// CHECK:STDOUT:   %HoldsType.loc18_46.1: type = class_type @HoldsType, @HoldsType(%T.loc18_7.1) [symbolic = %HoldsType.loc18_46.1 (constants.%HoldsType.9dd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %HoldsType.loc18_46.1 [symbolic = %pattern_type (constants.%pattern_type.dc7)]
// CHECK:STDOUT:   %x.param_patt.loc18_33.2: @F.%pattern_type (%pattern_type.dc7) = value_param_pattern [symbolic = %x.param_patt.loc18_33.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc18_33.2: @F.%pattern_type (%pattern_type.dc7) = wrapper_binding_pattern x, %x.param_patt.loc18_33.2 [symbolic = %x.patt.loc18_33.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HoldsType.loc18_46.1 [symbolic = %require_complete (constants.%require_complete.501)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%HoldsType.loc18_46.1 (%HoldsType.9dd), %a.param: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc23_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %HoldsType.ref: %HoldsType.type = name_ref HoldsType, file.%HoldsType.decl [concrete = constants.%HoldsType.generic]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc23_25.1: %tuple.type.135 = tuple_literal (%C.ref) [concrete = constants.%tuple.9e1]
// CHECK:STDOUT:   %.loc23_36: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Array: type = class_type @Array, @Array(type, constants.%int_1) [concrete = constants.%Array.612]
// CHECK:STDOUT:   %impl.elem0: %.cd2 = impl_witness_access constants.%Copy.impl_witness.f3e, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %C.ref, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%C.ref) [concrete = constants.%C]
// CHECK:STDOUT:   %.loc23_25.2: ref %Array.612 = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc23_25.3: ref type = array_index %.loc23_25.2, %int_0
// CHECK:STDOUT:   %.loc23_25.4: init type to %.loc23_25.3 = in_place_init %type.as.Copy.impl.Op.call [concrete = constants.%C]
// CHECK:STDOUT:   %.loc23_25.5: init %Array.612 to %.loc23_25.2 = array_init (%.loc23_25.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc23_27.1: init %Array.612 = converted %.loc23_25.1, %.loc23_25.5 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc23_27.2: ref %Array.612 = temporary %.loc23_25.2, %.loc23_27.1 [concrete = constants.%.345]
// CHECK:STDOUT:   %.loc23_27.3: %Array.612 = acquire_value %.loc23_27.2 [concrete = constants.%array]
// CHECK:STDOUT:   %HoldsType: type = class_type @HoldsType, @HoldsType(constants.%array) [concrete = constants.%HoldsType.00d]
// CHECK:STDOUT:   %.loc23_6.2: ref %HoldsType.00d = temporary_storage
// CHECK:STDOUT:   %.loc23_6.3: init %HoldsType.00d to %.loc23_6.2 = class_init () [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc23_8: init %HoldsType.00d = converted %.loc23_6.1, %.loc23_6.3 [concrete = constants.%HoldsType.val]
// CHECK:STDOUT:   %.loc23_48: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Destroy.facet.loc23_27.1: %Destroy.type = facet_value constants.%Array.612, (constants.%DestroyImpl.impl_witness.e45, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc23_27.4: %Destroy.type = converted constants.%Array.612, %Destroy.facet.loc23_27.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc23_27.2: %Destroy.type = facet_value constants.%Array.612, (constants.%DestroyImpl.impl_witness.e45, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc23_27.5: %Destroy.type = converted constants.%Array.612, %Destroy.facet.loc23_27.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc23_27.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23(%self.param: ref %Array.612) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%T.6be) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.bb1
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T.6be
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.6be) {
// CHECK:STDOUT:   %T.patt.loc18_7.2 => constants.%T.patt.bb1
// CHECK:STDOUT:   %T.loc18_7.1 => constants.%T.6be
// CHECK:STDOUT:   %HoldsType.loc18_46.1 => constants.%HoldsType.9dd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc7
// CHECK:STDOUT:   %x.param_patt.loc18_33.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc18_33.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HoldsType(constants.%array) {
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt.bb1
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%array
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
