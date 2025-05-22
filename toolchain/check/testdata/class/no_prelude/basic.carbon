// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/basic.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class {
  fn F(n: ()) -> () {
    return n;
  }

  fn G(n: {}) -> {};

  var k: {};
}

fn Class.G(n: {}) -> {} {
  return n;
}

fn Run() -> () {
  return Class.F(());
}
//@dump-sem-ir-end

// --- complete_in_member_fn.carbon
library "[[@TEST_NAME]]";

class C {
  fn F(c: C) -> () {
    //@dump-sem-ir-begin
    return c.a;
    //@dump-sem-ir-end
  }

  var a: ();
}

// --- fail_unknown_member.carbon
library "[[@TEST_NAME]]";

class Class {
  var n: ();
}

fn G(c: Class) -> () {
  // TODO: Mention the scope in which we looked for the name.
  // CHECK:STDERR: fail_unknown_member.carbon:[[@LINE+4]]:10: error: member name `something` not found in `Class` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return c.something;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  return c.something;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.k.df9: type = struct_type {.k: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k.df9 [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %n.patt: %pattern_type.a96 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.a96 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_23.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_23.2: type = converted %.loc14_23.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %n.param.loc14: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc14_16.1: type = splice_block %.loc14_16.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc14_16.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc14_16.3: type = converted %.loc14_16.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc14: %empty_struct_type = bind_name n, %n.param.loc14
// CHECK:STDOUT:     %return.param.loc14: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return.loc14: ref %empty_struct_type = return_slot %return.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_14.2: type = converted %.loc18_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %n.patt: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.cb1 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_19.2: type = converted %.loc5_19.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc5_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %n.patt: %pattern_type.a96 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.a96 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_19.2: type = converted %.loc9_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %n.param.loc9: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc9_12.1: type = splice_block %.loc9_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc9_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_12.3: type = converted %.loc9_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc9: %empty_struct_type = bind_name n, %n.param.loc9
// CHECK:STDOUT:     %return.param.loc9: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return.loc9: ref %empty_struct_type = return_slot %return.param.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_11.2: type = converted %.loc11_11.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc11_8: %Class.elem = field_decl k, element0 [concrete]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %empty_struct_type} [concrete = constants.%struct_type.k.df9]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .k = %.loc11_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param.loc14: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_struct_type = name_ref n, %n.loc14
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc19_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_19.2: %empty_tuple.type = converted %.loc19_19.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc19_19.2)
// CHECK:STDOUT:   %.loc19_20.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc19_20.2: ref %empty_tuple.type = temporary %.loc19_20.1, %F.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21: %empty_tuple.type = converted %F.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc19_21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- complete_in_member_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .a = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %a.ref: %C.elem = name_ref a, @C.%.loc10_8 [concrete = @C.%.loc10_8]
// CHECK:STDOUT:   %.loc6_13.1: ref %empty_tuple.type = class_element_access %c.ref, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_13.2: %empty_tuple.type = converted %.loc6_13.1, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc6_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
