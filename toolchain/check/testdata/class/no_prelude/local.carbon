// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/local.carbon

//@dump-sem-ir-begin
class A {
  fn F() -> () {
    class B {
      fn Make() -> Self {
        returned var b: Self = {.n = ()};
        return var;
      }

      var n: ();
    }

    return B.Make().n;
  }
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.971: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.880: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_14.2: type = converted %.loc15_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.971 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.971 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_15.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_15.2: type = converted %.loc22_15.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc22_12: %B.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %empty_tuple.type} [concrete = constants.%struct_type.n]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.880]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .n = %.loc22_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @B.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc25_19.1: ref %B = temporary_storage
// CHECK:STDOUT:   %Make.call: init %B = call %Make.ref() to %.loc25_19.1
// CHECK:STDOUT:   %.loc25_19.2: ref %B = temporary %.loc25_19.1, %Make.call
// CHECK:STDOUT:   %n.ref: %B.elem = name_ref n, @B.%.loc22_12 [concrete = @B.%.loc22_12]
// CHECK:STDOUT:   %.loc25_20: ref %empty_tuple.type = class_element_access %.loc25_19.2, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_22: %empty_tuple.type = converted %.loc25_20, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc25_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.971 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.971 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_39.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc18_40.1: %struct_type.n = struct_literal (%.loc18_39.1)
// CHECK:STDOUT:   %.loc18_40.2: ref %empty_tuple.type = class_element_access %return, element0
// CHECK:STDOUT:   %.loc18_39.2: init %empty_tuple.type = tuple_init () to %.loc18_40.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_40.3: init %empty_tuple.type = converted %.loc18_39.1, %.loc18_39.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_40.4: init %B = class_init (%.loc18_40.3), %return [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc18_18: init %B = converted %.loc18_40.1, %.loc18_40.4 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %return, %.loc18_18
// CHECK:STDOUT:   %Self.ref.loc18: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %return
// CHECK:STDOUT:   return %b to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
