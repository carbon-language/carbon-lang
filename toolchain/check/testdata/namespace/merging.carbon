// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/merging.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/merging.carbon

// --- a.carbon

package Example library "[[@TEST_NAME]]";

namespace NS;

fn NS.A() {}

// --- b.carbon

package Example library "[[@TEST_NAME]]";

namespace NS;

fn NS.B1() {}

namespace NS;

fn NS.B2() {}

// --- c.carbon

package Example library "[[@TEST_NAME]]";

import library "a";
import library "b";

namespace NS;

fn NS.C() {}

fn Run() {
  NS.A();
  NS.B1();
  NS.B2();
  NS.C();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1.type: type = fn_type @B1 [concrete]
// CHECK:STDOUT:   %B1: %B1.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.type: type = fn_type @B2 [concrete]
// CHECK:STDOUT:   %B2: %B2.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .NS = %NS.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS.loc4: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B1 = %B1.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B1.decl: %B1.type = fn_decl @B1 [concrete = constants.%B1] {} {}
// CHECK:STDOUT:   %NS.loc8: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B1 = %B1.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B2.decl: %B2.type = fn_decl @B2 [concrete = constants.%B2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.type: type = fn_type @B1 [concrete]
// CHECK:STDOUT:   %B1: %B1.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.type: type = fn_type @B2 [concrete]
// CHECK:STDOUT:   %B2: %B2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Example.NS: <namespace> = import_ref Example//a, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Example.NS, [concrete] {
// CHECK:STDOUT:     .A = %Example.A
// CHECK:STDOUT:     .B1 = %Example.B1
// CHECK:STDOUT:     .B2 = %Example.B2
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Example.A: %A.type = import_ref Example//a, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Example.B1: %B1.type = import_ref Example//b, B1, loaded [concrete = constants.%B1]
// CHECK:STDOUT:   %Example.B2: %B2.type = import_ref Example//b, B2, loaded [concrete = constants.%B2]
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
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Example.A
// CHECK:STDOUT:     .B1 = imports.%Example.B1
// CHECK:STDOUT:     .B2 = imports.%Example.B2
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref.loc12: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Example.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %B1.ref: %B1.type = name_ref B1, imports.%Example.B1 [concrete = constants.%B1]
// CHECK:STDOUT:   %B1.call: init %empty_tuple.type = call %B1.ref()
// CHECK:STDOUT:   %NS.ref.loc14: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %B2.ref: %B2.type = name_ref B2, imports.%Example.B2 [concrete = constants.%B2]
// CHECK:STDOUT:   %B2.call: init %empty_tuple.type = call %B2.ref()
// CHECK:STDOUT:   %NS.ref.loc15: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.call: init %empty_tuple.type = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B1() [from "b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B2() [from "b.carbon"];
// CHECK:STDOUT:
