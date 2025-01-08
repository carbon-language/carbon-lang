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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [template]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.a1d: type = class_type @C, @C(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %C.ec9: type = class_type @C, @C(%true) [template]
// CHECK:STDOUT:   %True.type: type = fn_type @True [template]
// CHECK:STDOUT:   %True: %True.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %C.9f8: type = class_type @C, @C(%false) [template]
// CHECK:STDOUT:   %False.type: type = fn_type @False [template]
// CHECK:STDOUT:   %False: %False.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %Eq.decl: %Eq.type = fn_decl @Eq [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: bool = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc4_28: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_28.1: type = value_of_initializer %bool.make_type.loc4_28 [template = bool]
// CHECK:STDOUT:     %.loc4_28.2: type = converted %bool.make_type.loc4_28, %.loc4_28.1 [template = bool]
// CHECK:STDOUT:     %a.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_10: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_10.2: type = value_of_initializer %bool.make_type.loc4_10 [template = bool]
// CHECK:STDOUT:       %.loc4_10.3: type = converted %bool.make_type.loc4_10, %.loc4_10.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4_19: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_19.2: type = value_of_initializer %bool.make_type.loc4_19 [template = bool]
// CHECK:STDOUT:       %.loc4_19.3: type = converted %bool.make_type.loc4_19, %.loc4_19.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %B.patt.loc6_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc6_9.1, runtime_param<invalid> [symbolic = %B.patt.loc6_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc6_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc6_13.3: type = converted %bool.make_type, %.loc6_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc6_9.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: %True.type = fn_decl @True [template = constants.%True] {
// CHECK:STDOUT:     %return.patt: %C.ec9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.ec9 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%true) [template = constants.%C.ec9]
// CHECK:STDOUT:     %return.param: ref %C.ec9 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.ec9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %False.decl: %False.type = fn_decl @False [template = constants.%False] {
// CHECK:STDOUT:     %return.patt: %C.9f8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.9f8 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%false) [template = constants.%C.9f8]
// CHECK:STDOUT:     %return.param: ref %C.9f8 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.9f8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.ec9 = var a
// CHECK:STDOUT:   %a: ref %C.ec9 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %C.9f8 = var b
// CHECK:STDOUT:   %b: ref %C.9f8 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %C.9f8 = var c
// CHECK:STDOUT:   %c: ref %C.9f8 = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref %C.ec9 = var d
// CHECK:STDOUT:   %d: ref %C.ec9 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%B.loc6_9.1: bool) {
// CHECK:STDOUT:   %B.loc6_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_9.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.a1d
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Eq(%a.param_patt: bool, %b.param_patt: bool) -> bool = "bool.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @True() -> %C.ec9;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @False() -> %C.9f8;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc11: %True.type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   %.loc11: ref %C.ec9 = splice_block file.%a.var {}
// CHECK:STDOUT:   %True.call.loc11: init %C.ec9 = call %True.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc11
// CHECK:STDOUT:   %False.ref.loc12: %False.type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   %.loc12: ref %C.9f8 = splice_block file.%b.var {}
// CHECK:STDOUT:   %False.call.loc12: init %C.9f8 = call %False.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%b.var, %False.call.loc12
// CHECK:STDOUT:   %False.ref.loc13: %False.type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   %.loc13: ref %C.9f8 = splice_block file.%c.var {}
// CHECK:STDOUT:   %False.call.loc13: init %C.9f8 = call %False.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign file.%c.var, %False.call.loc13
// CHECK:STDOUT:   %True.ref.loc14: %True.type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   %.loc14: ref %C.ec9 = splice_block file.%d.var {}
// CHECK:STDOUT:   %True.call.loc14: init %C.ec9 = call %True.ref.loc14() to %.loc14
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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.a1d: type = class_type @C, @C(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %C.ec9: type = class_type @C, @C(%true) [template]
// CHECK:STDOUT:   %True.type: type = fn_type @True [template]
// CHECK:STDOUT:   %True: %True.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %C.9f8: type = class_type @C, @C(%false) [template]
// CHECK:STDOUT:   %False.type: type = fn_type @False [template]
// CHECK:STDOUT:   %False: %False.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .Eq = %import_ref.32e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %B.patt.loc4_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc4_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc4_9.1, runtime_param<invalid> [symbolic = %B.patt.loc4_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc4_13.3: type = converted %bool.make_type, %.loc4_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc4_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc4_9.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: %True.type = fn_decl @True [template = constants.%True] {
// CHECK:STDOUT:     %return.patt: %C.ec9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.ec9 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%true) [template = constants.%C.ec9]
// CHECK:STDOUT:     %return.param: ref %C.ec9 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.ec9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %False.decl: %False.type = fn_decl @False [template = constants.%False] {
// CHECK:STDOUT:     %return.patt: %C.9f8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.9f8 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%false) [template = constants.%C.9f8]
// CHECK:STDOUT:     %return.param: ref %C.9f8 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.9f8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.ec9 = var a
// CHECK:STDOUT:   %a: ref %C.ec9 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %C.9f8 = var b
// CHECK:STDOUT:   %b: ref %C.9f8 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %C.9f8 = var c
// CHECK:STDOUT:   %c: ref %C.9f8 = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref %C.ec9 = var d
// CHECK:STDOUT:   %d: ref %C.ec9 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%B.loc4_9.1: bool) {
// CHECK:STDOUT:   %B.loc4_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc4_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc4_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc4_9.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.a1d
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @True() -> %C.ec9;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @False() -> %C.9f8;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc9: %True.type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   %.loc9: ref %C.ec9 = splice_block file.%a.var {}
// CHECK:STDOUT:   %True.call.loc9: init %C.ec9 = call %True.ref.loc9() to %.loc9
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc9
// CHECK:STDOUT:   %False.ref.loc10: %False.type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   %.loc10: ref %C.9f8 = splice_block file.%b.var {}
// CHECK:STDOUT:   %False.call.loc10: init %C.9f8 = call %False.ref.loc10() to %.loc10
// CHECK:STDOUT:   assign file.%b.var, %False.call.loc10
// CHECK:STDOUT:   %False.ref.loc11: %False.type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   %.loc11: ref %C.9f8 = splice_block file.%c.var {}
// CHECK:STDOUT:   %False.call.loc11: init %C.9f8 = call %False.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign file.%c.var, %False.call.loc11
// CHECK:STDOUT:   %True.ref.loc12: %True.type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   %.loc12: ref %C.ec9 = splice_block file.%d.var {}
// CHECK:STDOUT:   %True.call.loc12: init %C.ec9 = call %True.ref.loc12() to %.loc12
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
