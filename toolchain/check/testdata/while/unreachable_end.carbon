// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/unreachable_end.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/while/unreachable_end.carbon

fn Cond() -> bool;

fn F();
fn G();
fn H();

fn While() {
  F();
  while (Cond()) {
    G();
    return;
  }
  H();
}

// CHECK:STDOUT: --- unreachable_end.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Bool = struct_value () [template]
// CHECK:STDOUT:   %Cond: type = fn_type @Cond [template]
// CHECK:STDOUT:   %struct.2: Cond = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.4: G = struct_value () [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.5: H = struct_value () [template]
// CHECK:STDOUT:   %While: type = fn_type @While [template]
// CHECK:STDOUT:   %struct.6: While = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Cond = %Cond.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Cond.decl: Cond = fn_decl @Cond [template = constants.%struct.2] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %bool.make_type, %.loc11_14.1 [template = bool]
// CHECK:STDOUT:     @Cond.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.4] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.5] {}
// CHECK:STDOUT:   %While.decl: While = fn_decl @While [template = constants.%struct.6] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Cond() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %F.call: init () = call %F.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: Cond = name_ref Cond, file.%Cond.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Cond.call: init bool = call %Cond.ref()
// CHECK:STDOUT:   %.loc19_16.1: bool = value_of_initializer %Cond.call
// CHECK:STDOUT:   %.loc19_16.2: bool = converted %Cond.call, %.loc19_16.1
// CHECK:STDOUT:   if %.loc19_16.2 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %G.call: init () = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: H = name_ref H, file.%H.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %H.call: init () = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
