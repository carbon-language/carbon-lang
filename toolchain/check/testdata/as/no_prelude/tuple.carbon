// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/no_prelude/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/no_prelude/tuple.carbon

class X {
  // ...
}

fn Make() -> X;

fn Let() {
  // This should create value bindings for both tuple elements.
  let a: (X, X) = (Make(), Make()) as (X, X);
}

fn Var() {
  // This should initialize both tuple elements in place.
  var b: (X, X) = (Make(), Make()) as (X, X);
}

// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [concrete]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.b67: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [concrete]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [concrete = constants.%Let] {} {}
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [concrete = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.b67 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc19_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_25.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc19_25: init %X = call %Make.ref.loc19_20() to %.loc19_25.1
// CHECK:STDOUT:   %Make.ref.loc19_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc19_33: init %X = call %Make.ref.loc19_28() to %.loc19_33.1
// CHECK:STDOUT:   %.loc19_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc19_25, %Make.call.loc19_33)
// CHECK:STDOUT:   %X.ref.loc19_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc19_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc19_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc19_40, %X.ref.loc19_43)
// CHECK:STDOUT:   %.loc19_44.2: type = converted %.loc19_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc19_15.1: type = splice_block %.loc19_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc19_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc19_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc19_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc19_11, %X.ref.loc19_14)
// CHECK:STDOUT:     %.loc19_15.3: type = converted %.loc19_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_25.2: ref %X = temporary %.loc19_25.1, %Make.call.loc19_25
// CHECK:STDOUT:   %.loc19_25.3: %X = bind_value %.loc19_25.2
// CHECK:STDOUT:   %.loc19_33.2: ref %X = temporary %.loc19_33.1, %Make.call.loc19_33
// CHECK:STDOUT:   %.loc19_33.3: %X = bind_value %.loc19_33.2
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc19_25.3, %.loc19_33.3)
// CHECK:STDOUT:   %.loc19_34.2: %tuple.type.b67 = converted %.loc19_34.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.b67 = bind_name a, %.loc19_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.b67 = binding_pattern b
// CHECK:STDOUT:     %.loc24_3.1: %tuple.type.b67 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.b67 = var b
// CHECK:STDOUT:   %Make.ref.loc24_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc24_25: init %X = call %Make.ref.loc24_20() to %tuple.elem0
// CHECK:STDOUT:   %Make.ref.loc24_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc24_33: init %X = call %Make.ref.loc24_28() to %tuple.elem1
// CHECK:STDOUT:   %.loc24_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc24_25, %Make.call.loc24_33)
// CHECK:STDOUT:   %X.ref.loc24_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc24_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc24_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc24_40, %X.ref.loc24_43)
// CHECK:STDOUT:   %.loc24_44.2: type = converted %.loc24_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc24_34.2: init %tuple.type.b67 = tuple_init (%Make.call.loc24_25, %Make.call.loc24_33) to %b.var
// CHECK:STDOUT:   %.loc24_3.2: init %tuple.type.b67 = converted %.loc24_34.1, %.loc24_34.2
// CHECK:STDOUT:   assign %b.var, %.loc24_3.2
// CHECK:STDOUT:   %.loc24_15.1: type = splice_block %.loc24_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc24_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc24_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc24_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc24_11, %X.ref.loc24_14)
// CHECK:STDOUT:     %.loc24_15.3: type = converted %.loc24_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.b67 = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
