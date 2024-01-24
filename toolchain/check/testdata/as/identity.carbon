// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// `expr as T` should behave the same as `expr` if `T` is the type of `expr`.

class X {
  // ...
}

fn Value(n: X) {
  let m: X = n as X;
}

fn Reference(p: X*) {
  let q: X* = &(*p as X);
}

fn Make() -> X;

fn Initializing() {
  var x: X = (Make() as X);
}

// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type X [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.X = %X.decl, .Value = %Value, .Reference = %Reference, .Make = %Make, .Initializing = %Initializing} [template]
// CHECK:STDOUT:   %X.decl = class_decl @X, ()
// CHECK:STDOUT:   %Value: <function> = fn_decl @Value [template]
// CHECK:STDOUT:   %Reference: <function> = fn_decl @Reference [template]
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template]
// CHECK:STDOUT:   %Initializing: <function> = fn_decl @Initializing [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n: X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc14_10: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %n.ref: X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc14_19: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %m: X = bind_name m, %n.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p: X*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc18_10: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc18_11: type = ptr_type X [template = constants.%.4]
// CHECK:STDOUT:   %p.ref: X* = name_ref p, %p
// CHECK:STDOUT:   %.loc18_17: ref X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc18_23: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %.loc18_15: X* = addr_of %.loc18_17
// CHECK:STDOUT:   %q: X* = bind_name q, %.loc18_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc24_10: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   %x.var: ref X = var x
// CHECK:STDOUT:   %x: ref X = bind_name x, %x.var
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc24_7: ref X = splice_block %x.var {}
// CHECK:STDOUT:   %.loc24_19: init X = call %Make.ref() to %.loc24_7
// CHECK:STDOUT:   %X.ref.loc24_25: type = name_ref X, constants.%X [template = constants.%X]
// CHECK:STDOUT:   assign %x.var, %.loc24_19
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
