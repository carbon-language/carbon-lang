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
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %Class [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Class, %.4 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.n: %.2, .next: %.4} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %MakeReorder.type: type = fn_type @MakeReorder [template]
// CHECK:STDOUT:   %MakeReorder: %MakeReorder.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.next: %.4, .n: %.2} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .MakeReorder = %MakeReorder.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %n.patt: %.2 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.2 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %next.patt: %.4 = binding_pattern next
// CHECK:STDOUT:     %next.param_patt: %.4 = value_param_pattern %next.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc16_12.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc16_12.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc16_12.3: type = converted %int.make_type_signed, %.loc16_12.2 [template = constants.%.2]
// CHECK:STDOUT:     %Class.ref.loc16_23: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16_28: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %Class.ref.loc16_34: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %n.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.2 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %.4 = value_param runtime_param1
// CHECK:STDOUT:     %next: %.4 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeReorder.decl: %MakeReorder.type = fn_decl @MakeReorder [template = constants.%MakeReorder] {
// CHECK:STDOUT:     %n.patt: %.2 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %.2 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %next.patt: %.4 = binding_pattern next
// CHECK:STDOUT:     %next.param_patt: %.4 = value_param_pattern %next.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc20_19.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc20_19.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc20_19.3: type = converted %int.make_type_signed, %.loc20_19.2 [template = constants.%.2]
// CHECK:STDOUT:     %Class.ref.loc20_30: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc20_35: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %Class.ref.loc20_41: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %n.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %n: %.2 = bind_name n, %n.param
// CHECK:STDOUT:     %next.param: %.4 = value_param runtime_param1
// CHECK:STDOUT:     %next: %.4 = bind_name next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc12_10.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed, %.loc12_10.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_8: %.3 = field_decl n, element0 [template]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc13_18: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_11: %.5 = field_decl next, element1 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT:   .next = %.loc13_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param_patt: %.2, %next.param_patt: %.4) -> %return: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %.4 = name_ref next, %next
// CHECK:STDOUT:   %.loc17_31.1: %.6 = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %.loc17_31.2: ref %.2 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc17_31.3: init %.2 = initialize_from %n.ref to %.loc17_31.2
// CHECK:STDOUT:   %.loc17_31.4: ref %.4 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc17_31.5: init %.4 = initialize_from %next.ref to %.loc17_31.4
// CHECK:STDOUT:   %.loc17_31.6: init %Class = class_init (%.loc17_31.3, %.loc17_31.5), %return
// CHECK:STDOUT:   %.loc17_32: init %Class = converted %.loc17_31.1, %.loc17_31.6
// CHECK:STDOUT:   return %.loc17_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param_patt: %.2, %next.param_patt: %.4) -> %return: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %.4 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %.2 = name_ref n, %n
// CHECK:STDOUT:   %.loc21_31.1: %.9 = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %.loc21_31.2: ref %.2 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc21_31.3: init %.2 = initialize_from %n.ref to %.loc21_31.2
// CHECK:STDOUT:   %.loc21_31.4: ref %.4 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc21_31.5: init %.4 = initialize_from %next.ref to %.loc21_31.4
// CHECK:STDOUT:   %.loc21_31.6: init %Class = class_init (%.loc21_31.3, %.loc21_31.5), %return
// CHECK:STDOUT:   %.loc21_32: init %Class = converted %.loc21_31.1, %.loc21_31.6
// CHECK:STDOUT:   return %.loc21_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
