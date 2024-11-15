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
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [template]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%X, %X) [template]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [template]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [template = constants.%Let] {} {}
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [template = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc19_11: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc19_14: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc19_15.1: %tuple.type.1 = tuple_literal (%X.ref.loc19_11, %X.ref.loc19_14)
// CHECK:STDOUT:   %.loc19_15.2: type = converted %.loc19_15.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %Make.ref.loc19_20: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc19_24.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc19_24: init %X = call %Make.ref.loc19_20() to %.loc19_24.1
// CHECK:STDOUT:   %Make.ref.loc19_28: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc19_32.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc19_32: init %X = call %Make.ref.loc19_28() to %.loc19_32.1
// CHECK:STDOUT:   %.loc19_34: %tuple.type.2 = tuple_literal (%Make.call.loc19_24, %Make.call.loc19_32)
// CHECK:STDOUT:   %X.ref.loc19_40: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc19_43: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc19_44.1: %tuple.type.1 = tuple_literal (%X.ref.loc19_40, %X.ref.loc19_43)
// CHECK:STDOUT:   %.loc19_44.2: type = converted %.loc19_44.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc19_24.2: ref %X = temporary %.loc19_24.1, %Make.call.loc19_24
// CHECK:STDOUT:   %.loc19_24.3: %X = bind_value %.loc19_24.2
// CHECK:STDOUT:   %.loc19_32.2: ref %X = temporary %.loc19_32.1, %Make.call.loc19_32
// CHECK:STDOUT:   %.loc19_32.3: %X = bind_value %.loc19_32.2
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc19_24.3, %.loc19_32.3)
// CHECK:STDOUT:   %.loc19_45: %tuple.type.2 = converted %.loc19_34, %tuple
// CHECK:STDOUT:   %a: %tuple.type.2 = bind_name a, %.loc19_45
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc24_11: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc24_14: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc24_15.1: %tuple.type.1 = tuple_literal (%X.ref.loc24_11, %X.ref.loc24_14)
// CHECK:STDOUT:   %.loc24_15.2: type = converted %.loc24_15.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %Make.ref.loc24_20: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc24_34.1: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc24_24: init %X = call %Make.ref.loc24_20() to %.loc24_34.1
// CHECK:STDOUT:   %Make.ref.loc24_28: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc24_34.2: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc24_32: init %X = call %Make.ref.loc24_28() to %.loc24_34.2
// CHECK:STDOUT:   %.loc24_34.3: %tuple.type.2 = tuple_literal (%Make.call.loc24_24, %Make.call.loc24_32)
// CHECK:STDOUT:   %X.ref.loc24_40: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc24_43: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc24_44.1: %tuple.type.1 = tuple_literal (%X.ref.loc24_40, %X.ref.loc24_43)
// CHECK:STDOUT:   %.loc24_44.2: type = converted %.loc24_44.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc24_34.4: init %tuple.type.2 = tuple_init (%Make.call.loc24_24, %Make.call.loc24_32) to %b.var
// CHECK:STDOUT:   %.loc24_45: init %tuple.type.2 = converted %.loc24_34.3, %.loc24_34.4
// CHECK:STDOUT:   assign %b.var, %.loc24_45
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
