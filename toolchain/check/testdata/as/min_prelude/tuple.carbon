// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
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
// CHECK:STDOUT:     %a.patt: %tuple.type.b67 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc10_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_25.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc10_25: init %X = call %Make.ref.loc10_20() to %.loc10_25.1
// CHECK:STDOUT:   %Make.ref.loc10_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc10_33: init %X = call %Make.ref.loc10_28() to %.loc10_33.1
// CHECK:STDOUT:   %.loc10_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc10_25, %Make.call.loc10_33)
// CHECK:STDOUT:   %X.ref.loc10_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc10_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc10_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc10_40, %X.ref.loc10_43)
// CHECK:STDOUT:   %.loc10_44.2: type = converted %.loc10_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc10_15.1: type = splice_block %.loc10_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc10_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc10_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc10_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc10_11, %X.ref.loc10_14)
// CHECK:STDOUT:     %.loc10_15.3: type = converted %.loc10_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_25.2: ref %X = temporary %.loc10_25.1, %Make.call.loc10_25
// CHECK:STDOUT:   %.loc10_25.3: %X = bind_value %.loc10_25.2
// CHECK:STDOUT:   %.loc10_33.2: ref %X = temporary %.loc10_33.1, %Make.call.loc10_33
// CHECK:STDOUT:   %.loc10_33.3: %X = bind_value %.loc10_33.2
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc10_25.3, %.loc10_33.3)
// CHECK:STDOUT:   %.loc10_34.2: %tuple.type.b67 = converted %.loc10_34.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.b67 = bind_name a, %.loc10_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.b67 = binding_pattern b
// CHECK:STDOUT:     %.loc15_3.1: %tuple.type.b67 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.b67 = var b
// CHECK:STDOUT:   %Make.ref.loc15_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc15_25: init %X = call %Make.ref.loc15_20() to %tuple.elem0
// CHECK:STDOUT:   %Make.ref.loc15_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc15_33: init %X = call %Make.ref.loc15_28() to %tuple.elem1
// CHECK:STDOUT:   %.loc15_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc15_25, %Make.call.loc15_33)
// CHECK:STDOUT:   %X.ref.loc15_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc15_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc15_40, %X.ref.loc15_43)
// CHECK:STDOUT:   %.loc15_44.2: type = converted %.loc15_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc15_34.2: init %tuple.type.b67 = tuple_init (%Make.call.loc15_25, %Make.call.loc15_33) to %b.var
// CHECK:STDOUT:   %.loc15_3.2: init %tuple.type.b67 = converted %.loc15_34.1, %.loc15_34.2
// CHECK:STDOUT:   assign %b.var, %.loc15_3.2
// CHECK:STDOUT:   %.loc15_15.1: type = splice_block %.loc15_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc15_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc15_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc15_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc15_11, %X.ref.loc15_14)
// CHECK:STDOUT:     %.loc15_15.3: type = converted %.loc15_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.b67 = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
