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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon

fn Not(b: bool) -> bool {
  return not b;
}

let not_true: bool = not true;
let not_false: bool = not false;

fn Constant() {
  var unused a: if not true then bool else () = ();
  var unused b: if not false then bool else () = true;
}

// CHECK:STDOUT: --- unary_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.4e2: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %Not.type: type = fn_type @Not [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Not: %Not.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.713: <witness> = impl_witness imports.%Copy.impl_witness_table.1ff [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.713) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4a4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.15e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4a4, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %not_true.patt: %pattern_type.831 = value_binding_pattern not_true [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %not_false.patt: %pattern_type.831 = value_binding_pattern not_false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %b.patt.454: %pattern_type.831 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.831 = var_pattern %b.patt.454 [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc23 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ff = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Not = %Not.decl
// CHECK:STDOUT:     .not_true = %not_true
// CHECK:STDOUT:     .not_false = %not_false
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Not.decl: %Not.type = fn_decl @Not [concrete = constants.%Not] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc15_20.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_20.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc15_20.2: Core.Form = init_form %.loc15_20.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc15_11: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_true.patt: %pattern_type.831 = value_binding_pattern not_true [concrete = constants.%not_true.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_true: bool = wrapper_binding not_true, @__global_init.%.loc19
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_false.patt: %pattern_type.831 = value_binding_pattern not_false [concrete = constants.%not_false.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_false: bool = wrapper_binding not_false, @__global_init.%.loc20
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b.param: bool) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %.loc16: bool = not %b.ref
// CHECK:STDOUT:   %impl.elem0: %.15e = impl_witness_access constants.%Copy.impl_witness.713, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc16, %impl.elem0
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%.loc16)
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt
// CHECK:STDOUT:   %.loc23_50.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_50.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_3: init %empty_tuple.type = converted %.loc23_50.1, %.loc23_50.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc23_3
// CHECK:STDOUT:   br !.loc23_24
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_24:
// CHECK:STDOUT:   %true.loc23: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc23_20: bool = not %true.loc23 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc23_20 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %.loc23_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %.loc23_45: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_39: type = converted %.loc23_45, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_17: type = block_arg !if.expr.result.loc23 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc23_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_15:
// CHECK:STDOUT:   %a: ref %empty_tuple.type = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = ref_binding_pattern b [concrete = constants.%b.patt.454]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.831 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var %b.var_patt
// CHECK:STDOUT:   %true.loc24: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.15e = impl_witness_access constants.%Copy.impl_witness.713, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true.loc24, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true.loc24) [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   br !.loc24_24
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_24:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc24_20: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc24_20 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %.loc24_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %.loc24_46: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc24_40: type = converted %.loc24_46, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_40)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_17: type = block_arg !if.expr.result.loc24 [concrete = bool]
// CHECK:STDOUT:   br !.loc24_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_15:
// CHECK:STDOUT:   %b: ref bool = wrapper_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc24: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc24: init %empty_tuple.type = call %Destroy.Op.bound.loc24(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc19: bool = not %true [concrete = constants.%false]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc20: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
