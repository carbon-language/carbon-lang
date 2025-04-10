// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init.carbon

class Class {
  var n: i32;
  var next: Class*;
}

fn Make(n: i32, next: Class*) -> Class {
  return {.n = n, .next = next};
}

fn MakeReorder(n: i32, next: Class*) -> Class {
  return {.next = next, .n = n};
}

// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem.c91: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem.0c0: type = unbound_element_type %Class, %ptr.e71 [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %i32, .next: %ptr.e71} [concrete]
// CHECK:STDOUT:   %complete_type.78f: <witness> = complete_type_witness %struct_type.n.next [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeReorder.type: type = fn_type @MakeReorder [concrete]
// CHECK:STDOUT:   %MakeReorder: %MakeReorder.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.next.n: type = struct_type {.next: %ptr.e71, .n: %i32} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .MakeReorder = %MakeReorder.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %next.patt: %ptr.e71 = binding_pattern next
// CHECK:STDOUT:     %next.param_patt: %ptr.e71 = value_param_pattern %next.patt, call_param1
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc16_34: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc16_12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc16_28: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc16_23: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc16_23 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeReorder.decl: %MakeReorder.type = fn_decl @MakeReorder [concrete = constants.%MakeReorder] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %next.patt: %ptr.e71 = binding_pattern next
// CHECK:STDOUT:     %next.param_patt: %ptr.e71 = value_param_pattern %next.patt, call_param1
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc20_41: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc20_19: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc20_35: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc20_30: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc20_30 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc12_8: %Class.elem.c91 = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %Class.elem.c91 = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc12: ref %Class.elem.c91 = var <none>
// CHECK:STDOUT:   %.loc13_11: %Class.elem.0c0 = field_decl next, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc13_3: %Class.elem.0c0 = var_pattern %.loc13_11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc13: ref %Class.elem.0c0 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.next [concrete = constants.%complete_type.78f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .next = %.loc13_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param_patt: %i32, %next.param_patt: %ptr.e71) -> %return.param_patt: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %.loc17_31.1: %struct_type.n.next = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %.loc17_31.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc17_31.3: init %i32 = initialize_from %n.ref to %.loc17_31.2
// CHECK:STDOUT:   %.loc17_31.4: ref %ptr.e71 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc17_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc17_31.4
// CHECK:STDOUT:   %.loc17_31.6: init %Class = class_init (%.loc17_31.3, %.loc17_31.5), %return
// CHECK:STDOUT:   %.loc17_32: init %Class = converted %.loc17_31.1, %.loc17_31.6
// CHECK:STDOUT:   return %.loc17_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param_patt: %i32, %next.param_patt: %ptr.e71) -> %return.param_patt: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc21_31.1: %struct_type.next.n = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %.loc21_31.2: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc21_31.3: init %i32 = initialize_from %n.ref to %.loc21_31.2
// CHECK:STDOUT:   %.loc21_31.4: ref %ptr.e71 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc21_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc21_31.4
// CHECK:STDOUT:   %.loc21_31.6: init %Class = class_init (%.loc21_31.3, %.loc21_31.5), %return
// CHECK:STDOUT:   %.loc21_32: init %Class = converted %.loc21_31.1, %.loc21_31.6
// CHECK:STDOUT:   return %.loc21_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
