// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface HasF {
  fn F();
}

class C {
  extend impl as HasF {
    fn F() {}
  }
}

fn G(c: C) {
  // TODO: These should do impl lookup. Because that's not implemented yet, they
  // refer to the interface members rather than to the impl members.
  // CHECK:STDERR: fail_todo_extend_impl.carbon:[[@LINE+3]]:3: ERROR: Value of type `<associated <function> in HasF>` is not callable.
  // CHECK:STDERR:   C.F();
  // CHECK:STDERR:   ^~~~
  C.F();
  // CHECK:STDERR: fail_todo_extend_impl.carbon:[[@LINE+3]]:3: ERROR: Value of type `<associated <function> in HasF>` is not callable.
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^~~~
  c.F();
}

// CHECK:STDOUT: --- fail_todo_extend_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @HasF [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @HasF, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in HasF> = assoc_entity element0, @HasF.%F [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %.6: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %HasF.decl = interface_decl @HasF {} [template = constants.%.1]
// CHECK:STDOUT:   %C.decl = class_decl @C {} [template = constants.%C]
// CHECK:STDOUT:   %G: <function> = fn_decl @G {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc17_6.1: C = param c
// CHECK:STDOUT:     @G.%c: C = bind_name c, %c.loc17_6.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 {} [template]
// CHECK:STDOUT:   %.loc8: <associated <function> in HasF> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %.loc8
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as HasF {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 {} [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %F.ref.loc23: <associated <function> in HasF> = name_ref F, @HasF.%.loc8 [template = constants.%.3]
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc27: <associated <function> in HasF> = name_ref F, @HasF.%.loc8 [template = constants.%.3]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
