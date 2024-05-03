// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %Make: type = fn_type @Make [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Make = struct_value () [template]
// CHECK:STDOUT:   %Let: type = fn_type @Let [template]
// CHECK:STDOUT:   %struct.2: Let = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (X, X) [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = tuple_type ({}*, {}*) [template]
// CHECK:STDOUT:   %.7: type = ptr_type ({}*, {}*) [template]
// CHECK:STDOUT:   %Var: type = fn_type @Var [template]
// CHECK:STDOUT:   %struct.3: Var = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %Make.decl: Make = fn_decl @Make [template = constants.%struct.1] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     @Make.%return: ref X = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: Let = fn_decl @Let [template = constants.%struct.2] {}
// CHECK:STDOUT:   %Var.decl: Var = fn_decl @Var [template = constants.%struct.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc15_11: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc15_14: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc15_15.1: (type, type) = tuple_literal (%X.ref.loc15_11, %X.ref.loc15_14)
// CHECK:STDOUT:   %.loc15_15.2: type = converted %.loc15_15.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %Make.ref.loc15_20: Make = name_ref Make, file.%Make.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc15_24.1: ref X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc15_24: init X = call %Make.ref.loc15_20() to %.loc15_24.1
// CHECK:STDOUT:   %Make.ref.loc15_28: Make = name_ref Make, file.%Make.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc15_32.1: ref X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc15_32: init X = call %Make.ref.loc15_28() to %.loc15_32.1
// CHECK:STDOUT:   %.loc15_34: (X, X) = tuple_literal (%Make.call.loc15_24, %Make.call.loc15_32)
// CHECK:STDOUT:   %X.ref.loc15_40: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc15_43: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc15_44.1: (type, type) = tuple_literal (%X.ref.loc15_40, %X.ref.loc15_43)
// CHECK:STDOUT:   %.loc15_44.2: type = converted %.loc15_44.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_24.2: ref X = temporary %.loc15_24.1, %Make.call.loc15_24
// CHECK:STDOUT:   %.loc15_24.3: X = bind_value %.loc15_24.2
// CHECK:STDOUT:   %.loc15_32.2: ref X = temporary %.loc15_32.1, %Make.call.loc15_32
// CHECK:STDOUT:   %.loc15_32.3: X = bind_value %.loc15_32.2
// CHECK:STDOUT:   %tuple: (X, X) = tuple_value (%.loc15_24.3, %.loc15_32.3)
// CHECK:STDOUT:   %.loc15_45: (X, X) = converted %.loc15_34, %tuple
// CHECK:STDOUT:   %a: (X, X) = bind_name a, %.loc15_45
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc20_11: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_14: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc20_15.1: (type, type) = tuple_literal (%X.ref.loc20_11, %X.ref.loc20_14)
// CHECK:STDOUT:   %.loc20_15.2: type = converted %.loc20_15.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %b.var: ref (X, X) = var b
// CHECK:STDOUT:   %b: ref (X, X) = bind_name b, %b.var
// CHECK:STDOUT:   %Make.ref.loc20_20: Make = name_ref Make, file.%Make.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_34.1: ref X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc20_24: init X = call %Make.ref.loc20_20() to %.loc20_34.1
// CHECK:STDOUT:   %Make.ref.loc20_28: Make = name_ref Make, file.%Make.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_34.2: ref X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc20_32: init X = call %Make.ref.loc20_28() to %.loc20_34.2
// CHECK:STDOUT:   %.loc20_34.3: (X, X) = tuple_literal (%Make.call.loc20_24, %Make.call.loc20_32)
// CHECK:STDOUT:   %X.ref.loc20_40: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_43: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc20_44.1: (type, type) = tuple_literal (%X.ref.loc20_40, %X.ref.loc20_43)
// CHECK:STDOUT:   %.loc20_44.2: type = converted %.loc20_44.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_34.4: init (X, X) = tuple_init (%Make.call.loc20_24, %Make.call.loc20_32) to %b.var
// CHECK:STDOUT:   %.loc20_45: init (X, X) = converted %.loc20_34.3, %.loc20_34.4
// CHECK:STDOUT:   assign %b.var, %.loc20_45
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
