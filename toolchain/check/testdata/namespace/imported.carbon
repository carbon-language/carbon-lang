// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/imported.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/imported.carbon

// --- implicit.carbon

package Implicit;

namespace NS;
namespace NS.ChildNS;

fn NS.A();
fn NS.ChildNS.B();

// --- implicit.impl.carbon

impl package Implicit;

var a: () = NS.A();
var b: () = NS.ChildNS.B();

var package_a: () = package.NS.A();
var package_b: () = package.NS.ChildNS.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ChildNS = %ChildNS
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ChildNS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.NS: <namespace> = import_ref Implicit//default, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Implicit.NS, [concrete] {
// CHECK:STDOUT:     .A = %Implicit.A
// CHECK:STDOUT:     .ChildNS = %ChildNS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.A: %A.type = import_ref Implicit//default, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Implicit.ChildNS: <namespace> = import_ref Implicit//default, ChildNS, loaded
// CHECK:STDOUT:   %ChildNS: <namespace> = namespace %Implicit.ChildNS, [concrete] {
// CHECK:STDOUT:     .B = %Implicit.B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.B: %B.type = import_ref Implicit//default, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .package_a = %package_a
// CHECK:STDOUT:     .package_b = %package_b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:     %.loc5_1: %empty_tuple.type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %.loc5_9.1: type = splice_block %.loc5_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_9.3: type = converted %.loc5_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %package_a.patt: %empty_tuple.type = binding_pattern package_a
// CHECK:STDOUT:     %.loc7_1: %empty_tuple.type = var_pattern %package_a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_a.var: ref %empty_tuple.type = var package_a
// CHECK:STDOUT:   %.loc7_17.1: type = splice_block %.loc7_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_17.3: type = converted %.loc7_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_a: ref %empty_tuple.type = bind_name package_a, %package_a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %package_b.patt: %empty_tuple.type = binding_pattern package_b
// CHECK:STDOUT:     %.loc8_1: %empty_tuple.type = var_pattern %package_b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_b.var: ref %empty_tuple.type = var package_b
// CHECK:STDOUT:   %.loc8_17.1: type = splice_block %.loc8_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_17.3: type = converted %.loc8_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_b: ref %empty_tuple.type = bind_name package_b, %package_b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref.loc4: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %A.ref.loc4: %A.type = name_ref A, imports.%Implicit.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call.loc4: init %empty_tuple.type = call %A.ref.loc4()
// CHECK:STDOUT:   assign file.%a.var, %A.call.loc4
// CHECK:STDOUT:   %NS.ref.loc5: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %ChildNS.ref.loc5: <namespace> = name_ref ChildNS, imports.%ChildNS [concrete = imports.%ChildNS]
// CHECK:STDOUT:   %B.ref.loc5: %B.type = name_ref B, imports.%Implicit.B [concrete = constants.%B]
// CHECK:STDOUT:   %B.call.loc5: init %empty_tuple.type = call %B.ref.loc5()
// CHECK:STDOUT:   assign file.%b.var, %B.call.loc5
// CHECK:STDOUT:   %package.ref.loc7: <namespace> = name_ref package, package [concrete = package]
// CHECK:STDOUT:   %NS.ref.loc7: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %A.ref.loc7: %A.type = name_ref A, imports.%Implicit.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call.loc7: init %empty_tuple.type = call %A.ref.loc7()
// CHECK:STDOUT:   assign file.%package_a.var, %A.call.loc7
// CHECK:STDOUT:   %package.ref.loc8: <namespace> = name_ref package, package [concrete = package]
// CHECK:STDOUT:   %NS.ref.loc8: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %ChildNS.ref.loc8: <namespace> = name_ref ChildNS, imports.%ChildNS [concrete = imports.%ChildNS]
// CHECK:STDOUT:   %B.ref.loc8: %B.type = name_ref B, imports.%Implicit.B [concrete = constants.%B]
// CHECK:STDOUT:   %B.call.loc8: init %empty_tuple.type = call %B.ref.loc8()
// CHECK:STDOUT:   assign file.%package_b.var, %B.call.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
