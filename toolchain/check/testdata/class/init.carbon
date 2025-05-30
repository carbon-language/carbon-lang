// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem.c91: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem.0c0: type = unbound_element_type %Class, %ptr.e71 [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %i32, .next: %ptr.e71} [concrete]
// CHECK:STDOUT:   %complete_type.78f: <witness> = complete_type_witness %struct_type.n.next [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.796 = binding_pattern next [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.796 = value_param_pattern %next.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc19_34: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc19_12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc19_28: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc19_23: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc19_23 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeReorder.decl: %MakeReorder.type = fn_decl @MakeReorder [concrete = constants.%MakeReorder] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.796 = binding_pattern next [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.796 = value_param_pattern %next.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc23_41: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc23_19: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc23_35: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc23_30: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc23_30 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15: %Class.elem.c91 = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   %.loc16: %Class.elem.0c0 = field_decl next, element1 [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %i32, .next: %ptr.e71} [concrete = constants.%struct_type.n.next]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.next [concrete = constants.%complete_type.78f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .n = %.loc15
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .next = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param: %i32, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %.loc20_31.1: %struct_type.n.next = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %.loc20_31.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc20_31.3: init %i32 = initialize_from %n.ref to %.loc20_31.2
// CHECK:STDOUT:   %.loc20_31.4: ref %ptr.e71 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc20_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc20_31.4
// CHECK:STDOUT:   %.loc20_31.6: init %Class = class_init (%.loc20_31.3, %.loc20_31.5), %return
// CHECK:STDOUT:   %.loc20_32: init %Class = converted %.loc20_31.1, %.loc20_31.6
// CHECK:STDOUT:   return %.loc20_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param: %i32, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc24_31.1: %struct_type.next.n = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %.loc24_31.2: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc24_31.3: init %i32 = initialize_from %n.ref to %.loc24_31.2
// CHECK:STDOUT:   %.loc24_31.4: ref %ptr.e71 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc24_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc24_31.4
// CHECK:STDOUT:   %.loc24_31.6: init %Class = class_init (%.loc24_31.3, %.loc24_31.5), %return
// CHECK:STDOUT:   %.loc24_32: init %Class = converted %.loc24_31.1, %.loc24_31.6
// CHECK:STDOUT:   return %.loc24_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
