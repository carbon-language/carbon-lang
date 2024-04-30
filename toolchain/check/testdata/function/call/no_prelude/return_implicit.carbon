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
// CHECK:STDOUT:   %MakeImplicitEmptyTuple: type = fn_type @MakeImplicitEmptyTuple [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: MakeImplicitEmptyTuple = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .MakeImplicitEmptyTuple = %MakeImplicitEmptyTuple.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.decl: MakeImplicitEmptyTuple = fn_decl @MakeImplicitEmptyTuple [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {}
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
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.ref: MakeImplicitEmptyTuple = name_ref MakeImplicitEmptyTuple, file.%MakeImplicitEmptyTuple.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.call: init () = call %MakeImplicitEmptyTuple.ref()
// CHECK:STDOUT:   assign %b.var, %MakeImplicitEmptyTuple.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
