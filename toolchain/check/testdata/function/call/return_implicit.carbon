// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn MakeImplicitEmptyTuple() {
}

fn Main() {
  var b: () = MakeImplicitEmptyTuple();
}

// CHECK:STDOUT: --- return_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.MakeImplicitEmptyTuple = %MakeImplicitEmptyTuple, .Main = %Main} [template]
// CHECK:STDOUT:   %MakeImplicitEmptyTuple: <function> = fn_decl @MakeImplicitEmptyTuple [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeImplicitEmptyTuple() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_11.2: type = converted %.loc11_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.ref: <function> = name_ref MakeImplicitEmptyTuple, file.%MakeImplicitEmptyTuple [template = file.%MakeImplicitEmptyTuple]
// CHECK:STDOUT:   %.loc11_37: init () = call %MakeImplicitEmptyTuple.ref()
// CHECK:STDOUT:   assign %b.var, %.loc11_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
