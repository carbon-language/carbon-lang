// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/import_interface_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/import_interface_decl.carbon

// --- a.carbon
library "[[@TEST_NAME]]";
interface A;

// --- a.impl.carbon
impl library "[[@TEST_NAME]]";

// --- fail_b.carbon
library "[[@TEST_NAME]]";
interface B {}
// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as B;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as B;

// --- b.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A;
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A = import_ref Main//a, A, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_17.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc7_7.2 as %B.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.B = import_ref Main//b, B, unloaded
// CHECK:STDOUT:   %Main.import_ref.420 = import_ref Main//b, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5c: type = import_ref Main//b, loc7_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Main.import_ref.171: type = import_ref Main//b, loc7_12, loaded [concrete = constants.%B.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_17.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B [from "fail_b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.420
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.e5c as imports.%Main.import_ref.171 [from "fail_b.carbon"];
// CHECK:STDOUT:
