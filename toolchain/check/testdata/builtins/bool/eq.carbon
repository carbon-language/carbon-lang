// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/bool/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/bool/eq.carbon

// --- builtin_call.carbon

library "[[@TEST_NAME]]";

fn Eq(a: bool, b: bool) -> bool = "bool.eq";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

var a: C(Eq(true, true)) = True();
var b: C(Eq(true, false)) = False();
var c: C(Eq(false, true)) = False();
var d: C(Eq(false, false)) = True();

// --- prelude.carbon

library "[[@TEST_NAME]]";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

var a: C(true == true) = True();
var b: C(true == false) = False();
var c: C(false == true) = False();
var d: C(false == false) = True();

// CHECK:STDOUT: --- builtin_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [concrete]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.342: type = class_type @C, @C(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %C.a14: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %C.2ba: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type = fn_decl @Eq [concrete = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: bool = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc4_28: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc4_28.1: type = value_of_initializer %bool.make_type.loc4_28 [concrete = bool]
// CHECK:STDOUT:     %.loc4_28.2: type = converted %bool.make_type.loc4_28, %.loc4_28.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_10: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_10.2: type = value_of_initializer %bool.make_type.loc4_10 [concrete = bool]
// CHECK:STDOUT:       %.loc4_10.3: type = converted %bool.make_type.loc4_10, %.loc4_10.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_19: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_19.2: type = value_of_initializer %bool.make_type.loc4_19 [concrete = bool]
// CHECK:STDOUT:       %.loc4_19.3: type = converted %bool.make_type.loc4_19, %.loc4_19.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %B.patt.loc6_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc6_13.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc6_13.3: type = converted %bool.make_type, %.loc6_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_9.1: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_9.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: %True.type = fn_decl @True [concrete = constants.%True] {
// CHECK:STDOUT:     %return.patt: %C.a14 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.a14 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:     %return.param: ref %C.a14 = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.a14 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %False.decl: %False.type = fn_decl @False [concrete = constants.%False] {
// CHECK:STDOUT:     %return.patt: %C.2ba = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.2ba = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:     %return.param: ref %C.2ba = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.2ba = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %C.a14 = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %C.a14 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a14 = var a
// CHECK:STDOUT:   %.loc11_24.1: type = splice_block %C.loc11 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc11: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc11: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %true.loc11_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc11_19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %bool.eq.loc11: init bool = call %Eq.ref.loc11(%true.loc11_13, %true.loc11_19) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc11_24.2: bool = value_of_initializer %bool.eq.loc11 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc11_24.3: bool = converted %bool.eq.loc11, %.loc11_24.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc11: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a14 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %C.2ba = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %C.2ba = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %C.2ba = var b
// CHECK:STDOUT:   %.loc12_25.1: type = splice_block %C.loc12 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc12: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %true.loc12: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %bool.eq.loc12: init bool = call %Eq.ref.loc12(%true.loc12, %false.loc12) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_25.2: bool = value_of_initializer %bool.eq.loc12 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_25.3: bool = converted %bool.eq.loc12, %.loc12_25.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %C.2ba = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.2ba = binding_pattern c
// CHECK:STDOUT:     %.loc13_1: %C.2ba = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.2ba = var c
// CHECK:STDOUT:   %.loc13_25.1: type = splice_block %C.loc13 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc13: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc13: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %false.loc13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %true.loc13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %bool.eq.loc13: init bool = call %Eq.ref.loc13(%false.loc13, %true.loc13) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc13_25.2: bool = value_of_initializer %bool.eq.loc13 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc13_25.3: bool = converted %bool.eq.loc13, %.loc13_25.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc13: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.2ba = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %C.a14 = binding_pattern d
// CHECK:STDOUT:     %.loc14_1: %C.a14 = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %C.a14 = var d
// CHECK:STDOUT:   %.loc14_26.1: type = splice_block %C.loc14 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc14: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc14: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %false.loc14_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %false.loc14_20: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %bool.eq.loc14: init bool = call %Eq.ref.loc14(%false.loc14_13, %false.loc14_20) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc14_26.2: bool = value_of_initializer %bool.eq.loc14 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc14_26.3: bool = converted %bool.eq.loc14, %.loc14_26.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc14: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C.a14 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%B.loc6_9.1: bool) {
// CHECK:STDOUT:   %B.loc6_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_9.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.342
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Eq(%a.param_patt: bool, %b.param_patt: bool) -> bool = "bool.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @True() -> %C.a14;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @False() -> %C.2ba;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc11: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc11: ref %C.a14 = splice_block file.%a.var {}
// CHECK:STDOUT:   %True.call.loc11: init %C.a14 = call %True.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc11
// CHECK:STDOUT:   %False.ref.loc12: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc12: ref %C.2ba = splice_block file.%b.var {}
// CHECK:STDOUT:   %False.call.loc12: init %C.2ba = call %False.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%b.var, %False.call.loc12
// CHECK:STDOUT:   %False.ref.loc13: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc13: ref %C.2ba = splice_block file.%c.var {}
// CHECK:STDOUT:   %False.call.loc13: init %C.2ba = call %False.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign file.%c.var, %False.call.loc13
// CHECK:STDOUT:   %True.ref.loc14: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc14: ref %C.a14 = splice_block file.%d.var {}
// CHECK:STDOUT:   %True.call.loc14: init %C.a14 = call %True.ref.loc14() to %.loc14
// CHECK:STDOUT:   assign file.%d.var, %True.call.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%true) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%false) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.342: type = class_type @C, @C(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %C.a14: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %C.2ba: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eq.type: type = facet_type <@Eq> [concrete]
// CHECK:STDOUT:   %Equal.type.79c: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.998, imports.%Core.import_ref.fb6) [concrete]
// CHECK:STDOUT:   %Eq.facet: %Eq.type = facet_value bool, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.75b: type = fn_type_with_self_type %Equal.type.79c, %Eq.facet [concrete]
// CHECK:STDOUT:   %Equal.type.aa3: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.6e2: %Equal.type.aa3 = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.bound.fe0: <bound method> = bound_method %true, %Equal.6e2 [concrete]
// CHECK:STDOUT:   %Equal.bound.d3c: <bound method> = bound_method %false, %Equal.6e2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Eq = %Core.Eq
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %B.patt.loc4_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc4_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_13.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc4_13.3: type = converted %bool.make_type, %.loc4_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc4_9.1: bool = bind_symbolic_name B, 0 [symbolic = %B.loc4_9.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: %True.type = fn_decl @True [concrete = constants.%True] {
// CHECK:STDOUT:     %return.patt: %C.a14 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.a14 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:     %return.param: ref %C.a14 = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.a14 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %False.decl: %False.type = fn_decl @False [concrete = constants.%False] {
// CHECK:STDOUT:     %return.patt: %C.2ba = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.2ba = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:     %return.param: ref %C.2ba = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.2ba = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %C.a14 = binding_pattern a
// CHECK:STDOUT:     %.loc9_1: %C.a14 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a14 = var a
// CHECK:STDOUT:   %.loc9_22.1: type = splice_block %C.loc9 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc9: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc9_10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc9_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem0.loc9: %.75b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:     %bound_method.loc9: <bound method> = bound_method %true.loc9_10, %impl.elem0.loc9 [concrete = constants.%Equal.bound.fe0]
// CHECK:STDOUT:     %bool.eq.loc9: init bool = call %bound_method.loc9(%true.loc9_10, %true.loc9_18) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc9_22.2: bool = value_of_initializer %bool.eq.loc9 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc9_22.3: bool = converted %bool.eq.loc9, %.loc9_22.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc9: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a14 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %C.2ba = binding_pattern b
// CHECK:STDOUT:     %.loc10_1: %C.2ba = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %C.2ba = var b
// CHECK:STDOUT:   %.loc10_23.1: type = splice_block %C.loc10 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc10: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %false.loc10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %impl.elem0.loc10: %.75b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:     %bound_method.loc10: <bound method> = bound_method %true.loc10, %impl.elem0.loc10 [concrete = constants.%Equal.bound.fe0]
// CHECK:STDOUT:     %bool.eq.loc10: init bool = call %bound_method.loc10(%true.loc10, %false.loc10) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc10_23.2: bool = value_of_initializer %bool.eq.loc10 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc10_23.3: bool = converted %bool.eq.loc10, %.loc10_23.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc10: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %C.2ba = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.2ba = binding_pattern c
// CHECK:STDOUT:     %.loc11_1: %C.2ba = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.2ba = var c
// CHECK:STDOUT:   %.loc11_23.1: type = splice_block %C.loc11 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc11: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false.loc11: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %true.loc11: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem0.loc11: %.75b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:     %bound_method.loc11: <bound method> = bound_method %false.loc11, %impl.elem0.loc11 [concrete = constants.%Equal.bound.d3c]
// CHECK:STDOUT:     %bool.eq.loc11: init bool = call %bound_method.loc11(%false.loc11, %true.loc11) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc11_23.2: bool = value_of_initializer %bool.eq.loc11 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc11_23.3: bool = converted %bool.eq.loc11, %.loc11_23.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc11: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.2ba = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %C.a14 = binding_pattern d
// CHECK:STDOUT:     %.loc12_1: %C.a14 = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %C.a14 = var d
// CHECK:STDOUT:   %.loc12_24.1: type = splice_block %C.loc12 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false.loc12_10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %false.loc12_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %impl.elem0.loc12: %.75b = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:     %bound_method.loc12: <bound method> = bound_method %false.loc12_10, %impl.elem0.loc12 [concrete = constants.%Equal.bound.d3c]
// CHECK:STDOUT:     %bool.eq.loc12: init bool = call %bound_method.loc12(%false.loc12_10, %false.loc12_19) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_24.2: bool = value_of_initializer %bool.eq.loc12 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_24.3: bool = converted %bool.eq.loc12, %.loc12_24.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C.a14 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%B.loc4_9.1: bool) {
// CHECK:STDOUT:   %B.loc4_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc4_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc4_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc4_9.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.342
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @True() -> %C.a14;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @False() -> %C.2ba;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc9: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc9: ref %C.a14 = splice_block file.%a.var {}
// CHECK:STDOUT:   %True.call.loc9: init %C.a14 = call %True.ref.loc9() to %.loc9
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc9
// CHECK:STDOUT:   %False.ref.loc10: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc10: ref %C.2ba = splice_block file.%b.var {}
// CHECK:STDOUT:   %False.call.loc10: init %C.2ba = call %False.ref.loc10() to %.loc10
// CHECK:STDOUT:   assign file.%b.var, %False.call.loc10
// CHECK:STDOUT:   %False.ref.loc11: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc11: ref %C.2ba = splice_block file.%c.var {}
// CHECK:STDOUT:   %False.call.loc11: init %C.2ba = call %False.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%c.var, %False.call.loc11
// CHECK:STDOUT:   %True.ref.loc12: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc12: ref %C.a14 = splice_block file.%d.var {}
// CHECK:STDOUT:   %True.call.loc12: init %C.a14 = call %True.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%d.var, %True.call.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%true) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%false) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
