// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon

// --- fail_non_constant_result.carbon

package FailNonConstantResult;

fn PartialConstant(x: bool) {
  // CHECK:STDERR:        ╭─┤ fail_non_constant_result.carbon:[[@LINE+7]]:17
  // CHECK:STDERR: ->     │   var unused a: if true and x then bool else ();
  // CHECK:STDERR:        ·                 ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                                   ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused a: if true and x then bool else ();
  // CHECK:STDERR:        ╭─┤ fail_non_constant_result.carbon:[[@LINE+7]]:17
  // CHECK:STDERR: ->     │   var unused b: if false or x then bool else ();
  // CHECK:STDERR:        ·                 ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                                   ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused b: if false or x then bool else ();
}

// --- fail_despite_known_result.carbon

package FailDespiteKnownResult;

fn KnownValueButNonConstantCondition(x: bool) {
  // We choose not to give these cases constant values, even though we could,
  // because they notionally involve evaluating a non-constant condition.
  // CHECK:STDERR:        ╭─┤ fail_despite_known_result.carbon:[[@LINE+7]]:17
  // CHECK:STDERR: ->     │   var unused c: if x and false then bool else ();
  // CHECK:STDERR:        ·                 ━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                 ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                                    ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused c: if x and false then bool else ();
  // CHECK:STDERR:        ╭─┤ fail_despite_known_result.carbon:[[@LINE+7]]:17
  // CHECK:STDERR: ->     │   var unused d: if x or true then bool else ();
  // CHECK:STDERR:        ·                 ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                ┗━━┫ error: cannot evaluate type expression
  // CHECK:STDERR:        ·                                   ┃        [TypeExprEvaluationFailure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused d: if x or true then bool else ();
}

// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref <error> = var_storage %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   br !.loc12_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc12_20:
// CHECK:STDOUT:   %true.loc12: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc12 br !and.rhs else br !and.result(%false.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref.loc12: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc12_25: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc12_25 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %.loc12_36: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc12(%.loc12_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %.loc12_47: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_41: type = converted %.loc12_47, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc12(%.loc12_41)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_17: type = block_arg !if.expr.result.loc12
// CHECK:STDOUT:   br !.loc12_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc12_15:
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref <error> = var_storage %b.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   br !.loc20_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_20:
// CHECK:STDOUT:   %false.loc20: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc20_26.1: bool = not %false.loc20 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc20_26.1 br !or.rhs else br !or.result(%true.loc20)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref.loc20: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref.loc20)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc20_26.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc20_26.2 br !if.expr.then.loc20 else br !if.expr.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc20:
// CHECK:STDOUT:   %.loc20_36: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc20:
// CHECK:STDOUT:   %.loc20_47: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_41: type = converted %.loc20_47, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_41)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc20:
// CHECK:STDOUT:   %.loc20_17: type = block_arg !if.expr.result.loc20
// CHECK:STDOUT:   br !.loc20_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_15:
// CHECK:STDOUT:   %b: <error> = wrapper_binding b, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: <error> = ref_binding_pattern b [concrete = <error>]
// CHECK:STDOUT:     %b.var_patt: <error> = var_pattern %b.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.type: type = fn_type @KnownValueButNonConstantCondition [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition: %KnownValueButNonConstantCondition.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .KnownValueButNonConstantCondition = %KnownValueButNonConstantCondition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.decl: %KnownValueButNonConstantCondition.type = fn_decl @KnownValueButNonConstantCondition [concrete = constants.%KnownValueButNonConstantCondition] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref <error> = var_storage %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   br !.loc14_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_20:
// CHECK:STDOUT:   %x.ref.loc14: bool = name_ref x, %x
// CHECK:STDOUT:   %false.loc14_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %x.ref.loc14 br !and.rhs else br !and.result(%false.loc14_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %false.loc14_26: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result(%false.loc14_26)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc14_22: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc14_22 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %.loc14_37: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %.loc14_48: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_42: type = converted %.loc14_48, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_42)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_17: type = block_arg !if.expr.result.loc14
// CHECK:STDOUT:   br !.loc14_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_15:
// CHECK:STDOUT:   %c: <error> = wrapper_binding c, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = ref_binding_pattern c [concrete = <error>]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref <error> = var_storage %d.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   br !.loc22_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_20:
// CHECK:STDOUT:   %x.ref.loc22: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc22_22.1: bool = not %x.ref.loc22
// CHECK:STDOUT:   %true.loc22_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc22_22.1 br !or.rhs else br !or.result(%true.loc22_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %true.loc22_25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result(%true.loc22_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc22_22.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc22_22.2 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %.loc22_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_46: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_40: type = converted %.loc22_46, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_40)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_17: type = block_arg !if.expr.result.loc22
// CHECK:STDOUT:   br !.loc22_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_15:
// CHECK:STDOUT:   %d: <error> = wrapper_binding d, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: <error> = ref_binding_pattern d [concrete = <error>]
// CHECK:STDOUT:     %d.var_patt: <error> = var_pattern %d.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
