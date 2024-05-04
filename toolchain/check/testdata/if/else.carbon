// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}
fn G() {}
fn H() {}

fn If(b: bool) {
  if (b) {
    F();
  } else {
    G();
  }
  H();
}

// CHECK:STDOUT: --- else.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.3: H = struct_value () [template]
// CHECK:STDOUT:   %If: type = fn_type @If [template]
// CHECK:STDOUT:   %struct.4: If = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .If = %If.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.3] {}
// CHECK:STDOUT:   %If.decl: If = fn_decl @If [template = constants.%struct.4] {
// CHECK:STDOUT:     %b.loc11_7.1: bool = param b
// CHECK:STDOUT:     @If.%b: bool = bind_name b, %b.loc11_7.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If(%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init () = call %F.ref()
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %G.call: init () = call %G.ref()
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %H.ref: H = name_ref H, file.%H.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %H.call: init () = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
