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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/or.carbon

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

fn Constant() {
  var unused a: if true or true then bool else () = true;
  var unused b: if true or false then bool else () = true;
  var unused c: if false or true then bool else () = true;
  var unused d: if false or false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var unused a: if true or x then bool else () = true;
}

// CHECK:STDOUT: --- or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.348: <witness> = impl_witness imports.%Copy.impl_witness_table.3cc [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.348) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6dd: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.86d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6dd, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Or.type: type = fn_type @Or [concrete]
// CHECK:STDOUT:   %Or: %Or.type = struct_value () [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc93 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc77 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.595: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc124, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cc = impl_witness_table (%Core.import_ref.595), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Or = %Or.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc6.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc6.2: Core.Form = init_form %.loc6.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc17.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc17.2: Core.Form = init_form %.loc17.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or.decl: %Or.type = fn_decl @Or [concrete = constants.%Or] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc28.2: Core.Form = init_form %.loc28.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc116: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc34.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc34.2: bool = converted %F.call, %.loc34.1
// CHECK:STDOUT:   %.loc34.3: bool = not %.loc34.2
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc34.3 br !or.rhs else br !or.result(%true)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc34.4: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc34.5: bool = converted %G.call, %.loc34.4
// CHECK:STDOUT:   br !or.result(%.loc34.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc34.6: bool = block_arg !or.result
// CHECK:STDOUT:   %impl.elem0: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc34.6, %impl.elem0
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%.loc34.6)
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc59: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc59: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc59: <bound method> = bound_method %true.loc59, %impl.elem0.loc59 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc59: init bool = call %bound_method.loc59(%true.loc59) [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %bool.as.Copy.impl.Op.call.loc59
// CHECK:STDOUT:   br !.loc50
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc50:
// CHECK:STDOUT:   %true.loc50: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc51.1: bool = not %true.loc50 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc51: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc51.1 br !or.rhs.loc51 else br !or.result.loc51(%true.loc51)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc51:
// CHECK:STDOUT:   %true.loc52: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc51(%true.loc52)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc51:
// CHECK:STDOUT:   %.loc51.2: bool = block_arg !or.result.loc51 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc51.2 br !if.expr.then.loc49 else br !if.expr.else.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc49:
// CHECK:STDOUT:   %.loc54: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc54)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc49:
// CHECK:STDOUT:   %.loc57: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc55: type = converted %.loc57, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc55)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc49:
// CHECK:STDOUT:   %.loc49: type = block_arg !if.expr.result.loc49 [concrete = bool]
// CHECK:STDOUT:   br !.loc47
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc47:
// CHECK:STDOUT:   %a: ref bool = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.831 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var %b.var_patt
// CHECK:STDOUT:   %true.loc75: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc75: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc75: <bound method> = bound_method %true.loc75, %impl.elem0.loc75 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc75: init bool = call %bound_method.loc75(%true.loc75) [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %bool.as.Copy.impl.Op.call.loc75
// CHECK:STDOUT:   br !.loc66
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc66:
// CHECK:STDOUT:   %true.loc66: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc67.1: bool = not %true.loc66 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc67: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc67.1 br !or.rhs.loc67 else br !or.result.loc67(%true.loc67)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc67:
// CHECK:STDOUT:   %false.loc68: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc67(%false.loc68)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc67:
// CHECK:STDOUT:   %.loc67.2: bool = block_arg !or.result.loc67 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc67.2 br !if.expr.then.loc65 else br !if.expr.else.loc65
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc65:
// CHECK:STDOUT:   %.loc70: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc65(%.loc70)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc65:
// CHECK:STDOUT:   %.loc73: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc71: type = converted %.loc73, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc65(%.loc71)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc65:
// CHECK:STDOUT:   %.loc65: type = block_arg !if.expr.result.loc65 [concrete = bool]
// CHECK:STDOUT:   br !.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc63:
// CHECK:STDOUT:   %b: ref bool = ref_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.831 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref bool = var %c.var_patt
// CHECK:STDOUT:   %true.loc91: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc91: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc91: <bound method> = bound_method %true.loc91, %impl.elem0.loc91 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call.loc91: init bool = call %bound_method.loc91(%true.loc91) [concrete = constants.%true]
// CHECK:STDOUT:   assign %c.var, %bool.as.Copy.impl.Op.call.loc91
// CHECK:STDOUT:   br !.loc82
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc82:
// CHECK:STDOUT:   %false.loc82: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc83.1: bool = not %false.loc82 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc83: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc83.1 br !or.rhs.loc83 else br !or.result.loc83(%true.loc83)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc83:
// CHECK:STDOUT:   %true.loc84: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc83(%true.loc84)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc83:
// CHECK:STDOUT:   %.loc83.2: bool = block_arg !or.result.loc83 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc83.2 br !if.expr.then.loc81 else br !if.expr.else.loc81
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc81:
// CHECK:STDOUT:   %.loc86: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc81(%.loc86)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc81:
// CHECK:STDOUT:   %.loc89: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc87: type = converted %.loc89, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc81(%.loc87)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc81:
// CHECK:STDOUT:   %.loc81: type = block_arg !if.expr.result.loc81 [concrete = bool]
// CHECK:STDOUT:   br !.loc79
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc79:
// CHECK:STDOUT:   %c: ref bool = ref_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt
// CHECK:STDOUT:   %.loc108.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc108.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc93: init %empty_tuple.type = converted %.loc108.1, %.loc108.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc93
// CHECK:STDOUT:   br !.loc98
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc98:
// CHECK:STDOUT:   %false.loc98: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc99.1: bool = not %false.loc98 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc99: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc99.1 br !or.rhs.loc99 else br !or.result.loc99(%true.loc99)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc99:
// CHECK:STDOUT:   %false.loc100: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc99(%false.loc100)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc99:
// CHECK:STDOUT:   %.loc99.2: bool = block_arg !or.result.loc99 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc99.2 br !if.expr.then.loc97 else br !if.expr.else.loc97
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc97:
// CHECK:STDOUT:   %.loc102: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc97(%.loc102)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc97:
// CHECK:STDOUT:   %.loc105: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc103: type = converted %.loc105, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc97(%.loc103)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc97:
// CHECK:STDOUT:   %.loc97: type = block_arg !if.expr.result.loc97 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc95
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc95:
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc93: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc93: init %empty_tuple.type = call %Destroy.Op.bound.loc93(%d.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc77: <bound method> = bound_method %c.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc77: init %empty_tuple.type = call %Destroy.Op.bound.loc77(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc61: init %empty_tuple.type = call %Destroy.Op.bound.loc61(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc45: init %empty_tuple.type = call %Destroy.Op.bound.loc45(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc93(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc77(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc133: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true.loc133, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true.loc133) [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   br !.loc124
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc124:
// CHECK:STDOUT:   %true.loc124: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc125.1: bool = not %true.loc124 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc125: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc125.1 br !or.rhs else br !or.result(%true.loc125)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc125.2: bool = block_arg !or.result [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc125.2 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc128: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc128)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc131: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc129: type = converted %.loc131, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc129)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc123: type = block_arg !if.expr.result [concrete = bool]
// CHECK:STDOUT:   br !.loc121
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc121:
// CHECK:STDOUT:   %a: ref bool = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
