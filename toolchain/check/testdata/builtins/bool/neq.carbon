// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/bool/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/bool/neq.carbon

// --- builtin_call.carbon

library "[[@TEST_NAME]]";

fn Neq(a: bool, b: bool) -> bool = "bool.neq";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

var a: C(Neq(true, true)) = False();
var b: C(Neq(true, false)) = True();
var c: C(Neq(false, true)) = True();
var d: C(Neq(false, false)) = False();

// --- prelude.carbon

library "[[@TEST_NAME]]";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

var a: C(true != true) = False();
var b: C(true != false) = True();
var c: C(false != true) = True();
var d: C(false != false) = False();

// CHECK:STDOUT: --- builtin_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [concrete]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Neq = %Neq.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Neq.decl: %Neq.type = fn_decl @Neq [concrete = constants.%Neq] {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: bool = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc4_29: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc4_29.1: type = value_of_initializer %bool.make_type.loc4_29 [concrete = bool]
// CHECK:STDOUT:     %.loc4_29.2: type = converted %bool.make_type.loc4_29, %.loc4_29.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4_11.1: type = splice_block %.loc4_11.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_11.2: type = value_of_initializer %bool.make_type.loc4_11 [concrete = bool]
// CHECK:STDOUT:       %.loc4_11.3: type = converted %bool.make_type.loc4_11, %.loc4_11.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_20: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_20.2: type = value_of_initializer %bool.make_type.loc4_20 [concrete = bool]
// CHECK:STDOUT:       %.loc4_20.3: type = converted %bool.make_type.loc4_20, %.loc4_20.2 [concrete = bool]
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
// CHECK:STDOUT:     %a.patt: %C.2ba = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %C.2ba = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.2ba = var a
// CHECK:STDOUT:   %.loc11_25.1: type = splice_block %C.loc11 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc11: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc11: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %true.loc11_14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc11_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %bool.neq.loc11: init bool = call %Neq.ref.loc11(%true.loc11_14, %true.loc11_20) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc11_25.2: bool = value_of_initializer %bool.neq.loc11 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc11_25.3: bool = converted %bool.neq.loc11, %.loc11_25.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc11: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.2ba = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %C.a14 = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %C.a14 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %C.a14 = var b
// CHECK:STDOUT:   %.loc12_26.1: type = splice_block %C.loc12 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc12: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %true.loc12: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %bool.neq.loc12: init bool = call %Neq.ref.loc12(%true.loc12, %false.loc12) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_26.2: bool = value_of_initializer %bool.neq.loc12 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_26.3: bool = converted %bool.neq.loc12, %.loc12_26.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %C.a14 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.a14 = binding_pattern c
// CHECK:STDOUT:     %.loc13_1: %C.a14 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.a14 = var c
// CHECK:STDOUT:   %.loc13_26.1: type = splice_block %C.loc13 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc13: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc13: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %false.loc13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %true.loc13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %bool.neq.loc13: init bool = call %Neq.ref.loc13(%false.loc13, %true.loc13) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc13_26.2: bool = value_of_initializer %bool.neq.loc13 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc13_26.3: bool = converted %bool.neq.loc13, %.loc13_26.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc13: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.a14 = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %C.2ba = binding_pattern d
// CHECK:STDOUT:     %.loc14_1: %C.2ba = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %C.2ba = var d
// CHECK:STDOUT:   %.loc14_27.1: type = splice_block %C.loc14 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc14: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc14: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %false.loc14_14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %false.loc14_21: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %bool.neq.loc14: init bool = call %Neq.ref.loc14(%false.loc14_14, %false.loc14_21) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc14_27.2: bool = value_of_initializer %bool.neq.loc14 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc14_27.3: bool = converted %bool.neq.loc14, %.loc14_27.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc14: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C.2ba = bind_name d, %d.var
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
// CHECK:STDOUT: fn @Neq(%a.param_patt: bool, %b.param_patt: bool) -> bool = "bool.neq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @True() -> %C.a14;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @False() -> %C.2ba;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %False.ref.loc11: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc11: ref %C.2ba = splice_block file.%a.var {}
// CHECK:STDOUT:   %False.call.loc11: init %C.2ba = call %False.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc11
// CHECK:STDOUT:   %True.ref.loc12: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc12: ref %C.a14 = splice_block file.%b.var {}
// CHECK:STDOUT:   %True.call.loc12: init %C.a14 = call %True.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%b.var, %True.call.loc12
// CHECK:STDOUT:   %True.ref.loc13: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc13: ref %C.a14 = splice_block file.%c.var {}
// CHECK:STDOUT:   %True.call.loc13: init %C.a14 = call %True.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign file.%c.var, %True.call.loc13
// CHECK:STDOUT:   %False.ref.loc14: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc14: ref %C.2ba = splice_block file.%d.var {}
// CHECK:STDOUT:   %False.call.loc14: init %C.2ba = call %False.ref.loc14() to %.loc14
// CHECK:STDOUT:   assign file.%d.var, %False.call.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%B.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%true) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%B.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%false) {
// CHECK:STDOUT:   %B.loc6_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_9.2 => constants.%B.patt
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
// CHECK:STDOUT:   %NotEqual.type.e6c: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.85b, imports.%Core.import_ref.67a) [concrete]
// CHECK:STDOUT:   %Eq.facet: %Eq.type = facet_value bool, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.d6e: type = fn_type_with_self_type %NotEqual.type.e6c, %Eq.facet [concrete]
// CHECK:STDOUT:   %NotEqual.type.c0e: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.bf4: %NotEqual.type.c0e = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.bound.542: <bound method> = bound_method %true, %NotEqual.bf4 [concrete]
// CHECK:STDOUT:   %NotEqual.bound.5a9: <bound method> = bound_method %false, %NotEqual.bf4 [concrete]
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
// CHECK:STDOUT:     %a.patt: %C.2ba = binding_pattern a
// CHECK:STDOUT:     %.loc9_1: %C.2ba = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.2ba = var a
// CHECK:STDOUT:   %.loc9_22.1: type = splice_block %C.loc9 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc9: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc9_10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc9_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem1.loc9: %.d6e = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%NotEqual.bf4]
// CHECK:STDOUT:     %bound_method.loc9: <bound method> = bound_method %true.loc9_10, %impl.elem1.loc9 [concrete = constants.%NotEqual.bound.542]
// CHECK:STDOUT:     %bool.neq.loc9: init bool = call %bound_method.loc9(%true.loc9_10, %true.loc9_18) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc9_22.2: bool = value_of_initializer %bool.neq.loc9 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc9_22.3: bool = converted %bool.neq.loc9, %.loc9_22.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc9: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.2ba = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %C.a14 = binding_pattern b
// CHECK:STDOUT:     %.loc10_1: %C.a14 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %C.a14 = var b
// CHECK:STDOUT:   %.loc10_23.1: type = splice_block %C.loc10 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc10: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %false.loc10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %impl.elem1.loc10: %.d6e = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%NotEqual.bf4]
// CHECK:STDOUT:     %bound_method.loc10: <bound method> = bound_method %true.loc10, %impl.elem1.loc10 [concrete = constants.%NotEqual.bound.542]
// CHECK:STDOUT:     %bool.neq.loc10: init bool = call %bound_method.loc10(%true.loc10, %false.loc10) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc10_23.2: bool = value_of_initializer %bool.neq.loc10 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc10_23.3: bool = converted %bool.neq.loc10, %.loc10_23.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc10: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %C.a14 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.a14 = binding_pattern c
// CHECK:STDOUT:     %.loc11_1: %C.a14 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.a14 = var c
// CHECK:STDOUT:   %.loc11_23.1: type = splice_block %C.loc11 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc11: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false.loc11: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %true.loc11: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem1.loc11: %.d6e = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%NotEqual.bf4]
// CHECK:STDOUT:     %bound_method.loc11: <bound method> = bound_method %false.loc11, %impl.elem1.loc11 [concrete = constants.%NotEqual.bound.5a9]
// CHECK:STDOUT:     %bool.neq.loc11: init bool = call %bound_method.loc11(%false.loc11, %true.loc11) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc11_23.2: bool = value_of_initializer %bool.neq.loc11 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc11_23.3: bool = converted %bool.neq.loc11, %.loc11_23.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc11: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.a14 = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %C.2ba = binding_pattern d
// CHECK:STDOUT:     %.loc12_1: %C.2ba = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %C.2ba = var d
// CHECK:STDOUT:   %.loc12_24.1: type = splice_block %C.loc12 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %false.loc12_10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %false.loc12_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %impl.elem1.loc12: %.d6e = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%NotEqual.bf4]
// CHECK:STDOUT:     %bound_method.loc12: <bound method> = bound_method %false.loc12_10, %impl.elem1.loc12 [concrete = constants.%NotEqual.bound.5a9]
// CHECK:STDOUT:     %bool.neq.loc12: init bool = call %bound_method.loc12(%false.loc12_10, %false.loc12_19) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_24.2: bool = value_of_initializer %bool.neq.loc12 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_24.3: bool = converted %bool.neq.loc12, %.loc12_24.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C.2ba = bind_name d, %d.var
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
// CHECK:STDOUT:   %False.ref.loc9: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc9: ref %C.2ba = splice_block file.%a.var {}
// CHECK:STDOUT:   %False.call.loc9: init %C.2ba = call %False.ref.loc9() to %.loc9
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc9
// CHECK:STDOUT:   %True.ref.loc10: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc10: ref %C.a14 = splice_block file.%b.var {}
// CHECK:STDOUT:   %True.call.loc10: init %C.a14 = call %True.ref.loc10() to %.loc10
// CHECK:STDOUT:   assign file.%b.var, %True.call.loc10
// CHECK:STDOUT:   %True.ref.loc11: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc11: ref %C.a14 = splice_block file.%c.var {}
// CHECK:STDOUT:   %True.call.loc11: init %C.a14 = call %True.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%c.var, %True.call.loc11
// CHECK:STDOUT:   %False.ref.loc12: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc12: ref %C.2ba = splice_block file.%d.var {}
// CHECK:STDOUT:   %False.call.loc12: init %C.2ba = call %False.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%d.var, %False.call.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%B.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%true) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%B.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%false) {
// CHECK:STDOUT:   %B.loc4_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc4_9.2 => constants.%B.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
