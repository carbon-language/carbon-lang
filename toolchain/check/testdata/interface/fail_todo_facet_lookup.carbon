// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface { fn F(); }

fn CallStatic(T:! Interface) {
  // CHECK:STDERR: fail_todo_facet_lookup.carbon:[[@LINE+3]]:3: ERROR: Type `Interface` does not support qualified expressions.
  // CHECK:STDERR:   T.F();
  // CHECK:STDERR:   ^~~
  T.F();
}

// CHECK:STDOUT: --- fail_todo_facet_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Interface = %Interface.decl, .CallStatic = %CallStatic} [template]
// CHECK:STDOUT:   %Interface.decl = interface_decl @Interface {} [template = constants.%.1]
// CHECK:STDOUT:   %CallStatic: <function> = fn_decl @CallStatic {
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, %Interface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc9_15.1: Interface = param T
// CHECK:STDOUT:     @CallStatic.%T: Interface = bind_symbolic_name T, %T.loc9_15.1 [symbolic]
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %F: <function> = fn_decl @F {} [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStatic(%T: Interface) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: Interface = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
