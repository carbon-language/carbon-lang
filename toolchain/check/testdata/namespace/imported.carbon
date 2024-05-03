// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

namespace NS;
namespace NS.ChildNS;

fn NS.A();
fn NS.ChildNS.B();

// --- implicit.impl.carbon

package Implicit impl;

var a: () = NS.A();
var b: () = NS.ChildNS.B();

var package_a: () = package.NS.A();
var package_b: () = package.NS.ChildNS.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ChildNS = %ChildNS
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ChildNS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .package_a = %package_a
// CHECK:STDOUT:     .package_b = %package_b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir0, inst+2, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .ChildNS = %ChildNS
// CHECK:STDOUT:     .A = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir0, inst+3, loaded
// CHECK:STDOUT:   %ChildNS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .B = %import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: A = import_ref ir0, inst+4, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: B = import_ref ir0, inst+8, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.loc5_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc5_9.2: type = converted %.loc5_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %.loc7_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_a.var: ref () = var package_a
// CHECK:STDOUT:   %package_a: ref () = bind_name package_a, %package_a.var
// CHECK:STDOUT:   %.loc8_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_17.2: type = converted %.loc8_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_b.var: ref () = var package_b
// CHECK:STDOUT:   %package_b: ref () = bind_name package_b, %package_b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref.loc4: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %A.ref.loc4: A = name_ref A, file.%import_ref.3 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call.loc4: init () = call %A.ref.loc4()
// CHECK:STDOUT:   assign file.%a.var, %A.call.loc4
// CHECK:STDOUT:   %NS.ref.loc5: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %ChildNS.ref.loc5: <namespace> = name_ref ChildNS, file.%ChildNS [template = file.%ChildNS]
// CHECK:STDOUT:   %B.ref.loc5: B = name_ref B, file.%import_ref.4 [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call.loc5: init () = call %B.ref.loc5()
// CHECK:STDOUT:   assign file.%b.var, %B.call.loc5
// CHECK:STDOUT:   %package.ref.loc7: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref.loc7: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %A.ref.loc7: A = name_ref A, file.%import_ref.3 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call.loc7: init () = call %A.ref.loc7()
// CHECK:STDOUT:   assign file.%package_a.var, %A.call.loc7
// CHECK:STDOUT:   %package.ref.loc8: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref.loc8: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %ChildNS.ref.loc8: <namespace> = name_ref ChildNS, file.%ChildNS [template = file.%ChildNS]
// CHECK:STDOUT:   %B.ref.loc8: B = name_ref B, file.%import_ref.4 [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call.loc8: init () = call %B.ref.loc8()
// CHECK:STDOUT:   assign file.%package_b.var, %B.call.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
