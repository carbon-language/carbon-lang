// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/min_prelude/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/min_prelude/tuple.carbon

// --- test.carbon

library "[[@TEST_NAME]]";

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

// CHECK:STDOUT: --- test.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [concrete]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.b67: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.bb7: type = pattern_type %tuple.type.b67 [concrete]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [concrete]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.019 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.019 = out_param_pattern %return.patt, call_param0 [concrete]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.bb7 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc12_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_25.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc12_25: init %X = call %Make.ref.loc12_20() to %.loc12_25.1
// CHECK:STDOUT:   %Make.ref.loc12_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc12_33: init %X = call %Make.ref.loc12_28() to %.loc12_33.1
// CHECK:STDOUT:   %.loc12_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc12_25, %Make.call.loc12_33)
// CHECK:STDOUT:   %X.ref.loc12_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc12_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc12_40, %X.ref.loc12_43)
// CHECK:STDOUT:   %.loc12_44.2: type = converted %.loc12_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc12_15.1: type = splice_block %.loc12_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc12_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc12_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc12_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc12_11, %X.ref.loc12_14)
// CHECK:STDOUT:     %.loc12_15.3: type = converted %.loc12_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_25.2: ref %X = temporary %.loc12_25.1, %Make.call.loc12_25
// CHECK:STDOUT:   %.loc12_25.3: %X = bind_value %.loc12_25.2
// CHECK:STDOUT:   %.loc12_33.2: ref %X = temporary %.loc12_33.1, %Make.call.loc12_33
// CHECK:STDOUT:   %.loc12_33.3: %X = bind_value %.loc12_33.2
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc12_25.3, %.loc12_33.3)
// CHECK:STDOUT:   %.loc12_34.2: %tuple.type.b67 = converted %.loc12_34.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.b67 = bind_name a, %.loc12_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.bb7 = binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc17_3.1: %pattern_type.bb7 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.b67 = var b
// CHECK:STDOUT:   %Make.ref.loc17_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc17_25: init %X = call %Make.ref.loc17_20() to %tuple.elem0
// CHECK:STDOUT:   %Make.ref.loc17_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc17_33: init %X = call %Make.ref.loc17_28() to %tuple.elem1
// CHECK:STDOUT:   %.loc17_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc17_25, %Make.call.loc17_33)
// CHECK:STDOUT:   %X.ref.loc17_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc17_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc17_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc17_40, %X.ref.loc17_43)
// CHECK:STDOUT:   %.loc17_44.2: type = converted %.loc17_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc17_34.2: init %tuple.type.b67 = tuple_init (%Make.call.loc17_25, %Make.call.loc17_33) to %b.var
// CHECK:STDOUT:   %.loc17_3.2: init %tuple.type.b67 = converted %.loc17_34.1, %.loc17_34.2
// CHECK:STDOUT:   assign %b.var, %.loc17_3.2
// CHECK:STDOUT:   %.loc17_15.1: type = splice_block %.loc17_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc17_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc17_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc17_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc17_11, %X.ref.loc17_14)
// CHECK:STDOUT:     %.loc17_15.3: type = converted %.loc17_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.b67 = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
