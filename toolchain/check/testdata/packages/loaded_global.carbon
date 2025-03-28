// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/loaded_global.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/loaded_global.carbon

// --- implicit.carbon

package Implicit;

fn A();

// --- implicit.impl.carbon

impl package Implicit;

var a: () = A();

var package_a: () = package.A();

// --- same_package.carbon

package SamePackage;

fn B();

// --- same_package_importer.carbon

package SamePackage library "[[@TEST_NAME]]";

import library default;

var b: () = B();

var package_b: () = package.B();

// CHECK:STDOUT: --- implicit.carbon
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.A: %A.type = import_ref Implicit//default, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Implicit.A
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .package_a = %package_a
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
// CHECK:STDOUT:     %package_a.patt: %empty_tuple.type = binding_pattern package_a
// CHECK:STDOUT:     %.loc6_1: %empty_tuple.type = var_pattern %package_a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_a.var: ref %empty_tuple.type = var package_a
// CHECK:STDOUT:   %.loc6_17.1: type = splice_block %.loc6_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_17.3: type = converted %.loc6_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %package_a: ref %empty_tuple.type = bind_name package_a, %package_a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc4: %A.type = name_ref A, imports.%Implicit.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call.loc4: init %empty_tuple.type = call %A.ref.loc4()
// CHECK:STDOUT:   assign file.%a.var, %A.call.loc4
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [concrete = package]
// CHECK:STDOUT:   %A.ref.loc6: %A.type = name_ref A, imports.%Implicit.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call.loc6: init %empty_tuple.type = call %A.ref.loc6()
// CHECK:STDOUT:   assign file.%package_a.var, %A.call.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package_importer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %SamePackage.B: %B.type = import_ref SamePackage//default, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = imports.%SamePackage.B
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .package_b = %package_b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:     %.loc6_1: %empty_tuple.type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_9.3: type = converted %.loc6_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
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
// CHECK:STDOUT: fn @B() [from "same_package.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %B.ref.loc6: %B.type = name_ref B, imports.%SamePackage.B [concrete = constants.%B]
// CHECK:STDOUT:   %B.call.loc6: init %empty_tuple.type = call %B.ref.loc6()
// CHECK:STDOUT:   assign file.%b.var, %B.call.loc6
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [concrete = package]
// CHECK:STDOUT:   %B.ref.loc8: %B.type = name_ref B, imports.%SamePackage.B [concrete = constants.%B]
// CHECK:STDOUT:   %B.call.loc8: init %empty_tuple.type = call %B.ref.loc8()
// CHECK:STDOUT:   assign file.%package_b.var, %B.call.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
