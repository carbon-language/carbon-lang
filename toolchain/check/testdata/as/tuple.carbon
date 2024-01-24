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
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (X, X) [template]
// CHECK:STDOUT:   %.6: type = tuple_type ({}*, {}*) [template]
// CHECK:STDOUT:   %.7: type = ptr_type ({}*, {}*) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.X = %X.decl, .Make = %Make, .Let = %Let, .Var = %Var} [template]
// CHECK:STDOUT:   %X.decl = class_decl @X, ()
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template]
// CHECK:STDOUT:   %Let: <function> = fn_decl @Let [template]
// CHECK:STDOUT:   %Var: <function> = fn_decl @Var [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc15_11: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc15_14: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc15_15.1: (type, type) = tuple_literal (%X.ref.loc15_11, %X.ref.loc15_14)
// CHECK:STDOUT:   %.loc15_15.2: type = converted %.loc15_15.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %Make.ref.loc15_20: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc15_24.1: ref X = temporary_storage
// CHECK:STDOUT:   %.loc15_24.2: init X = call %Make.ref.loc15_20() to %.loc15_24.1
// CHECK:STDOUT:   %Make.ref.loc15_28: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc15_32.1: ref X = temporary_storage
// CHECK:STDOUT:   %.loc15_32.2: init X = call %Make.ref.loc15_28() to %.loc15_32.1
// CHECK:STDOUT:   %.loc15_34.1: (X, X) = tuple_literal (%.loc15_24.2, %.loc15_32.2)
// CHECK:STDOUT:   %X.ref.loc15_40: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc15_43: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc15_44.1: (type, type) = tuple_literal (%X.ref.loc15_40, %X.ref.loc15_43)
// CHECK:STDOUT:   %.loc15_44.2: type = converted %.loc15_44.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_24.3: ref X = temporary %.loc15_24.1, %.loc15_24.2
// CHECK:STDOUT:   %.loc15_24.4: X = bind_value %.loc15_24.3
// CHECK:STDOUT:   %.loc15_32.3: ref X = temporary %.loc15_32.1, %.loc15_32.2
// CHECK:STDOUT:   %.loc15_32.4: X = bind_value %.loc15_32.3
// CHECK:STDOUT:   %.loc15_34.2: (X, X) = tuple_value (%.loc15_24.4, %.loc15_32.4)
// CHECK:STDOUT:   %.loc15_34.3: (X, X) = converted %.loc15_34.1, %.loc15_34.2
// CHECK:STDOUT:   %a: (X, X) = bind_name a, %.loc15_34.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc20_11: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_14: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc20_15.1: (type, type) = tuple_literal (%X.ref.loc20_11, %X.ref.loc20_14)
// CHECK:STDOUT:   %.loc20_15.2: type = converted %.loc20_15.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %b.var: ref (X, X) = var b
// CHECK:STDOUT:   %b: ref (X, X) = bind_name b, %b.var
// CHECK:STDOUT:   %Make.ref.loc20_20: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc20_34.1: ref X = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc20_24: init X = call %Make.ref.loc20_20() to %.loc20_34.1
// CHECK:STDOUT:   %Make.ref.loc20_28: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc20_34.2: ref X = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc20_32: init X = call %Make.ref.loc20_28() to %.loc20_34.2
// CHECK:STDOUT:   %.loc20_34.3: (X, X) = tuple_literal (%.loc20_24, %.loc20_32)
// CHECK:STDOUT:   %X.ref.loc20_40: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_43: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc20_44.1: (type, type) = tuple_literal (%X.ref.loc20_40, %X.ref.loc20_43)
// CHECK:STDOUT:   %.loc20_44.2: type = converted %.loc20_44.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc20_34.4: init (X, X) = tuple_init (%.loc20_24, %.loc20_32) to %b.var
// CHECK:STDOUT:   %.loc20_34.5: init (X, X) = converted %.loc20_34.3, %.loc20_34.4
// CHECK:STDOUT:   assign %b.var, %.loc20_34.5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
