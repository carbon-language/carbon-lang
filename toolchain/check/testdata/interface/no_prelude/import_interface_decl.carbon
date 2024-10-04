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
impl () as A;


// --- a.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = interface_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [template = constants.%A.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc3_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc3_7.2: type = converted %.loc3_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc3_7.2 as %A.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = interface_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//a, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+5, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//a, inst+6, loaded [template = constants.%A.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc1_6.2 = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.2 as imports.%import_ref.3;
// CHECK:STDOUT:
