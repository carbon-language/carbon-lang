// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_definition_in_impl_file.carbon

// --- a.carbon

library "a";

fn AForwardDecl();

// --- a.impl.carbon

impl library "a";

fn AForwardDecl();

fn AForwardDecl() {}

// --- b.carbon

library "b";

fn BForwardDecl();

// --- fail_b.impl.carbon

impl library "b";

import library "a";

// CHECK:STDERR: fail_b.impl.carbon:[[@LINE+3]]:1: ERROR: No definition found for declaration in impl file
// CHECK:STDERR: fn BForwardDecl();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
fn BForwardDecl();

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AForwardDecl.type: type = fn_type @AForwardDecl [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %AForwardDecl: %AForwardDecl.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .AForwardDecl = %AForwardDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %AForwardDecl.decl: %AForwardDecl.type = fn_decl @AForwardDecl [template = constants.%AForwardDecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AForwardDecl();
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AForwardDecl.type: type = fn_type @AForwardDecl [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %AForwardDecl: %AForwardDecl.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .AForwardDecl = %AForwardDecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %AForwardDecl.type = import_ref ir0, inst+2, loaded [template = constants.%AForwardDecl]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %AForwardDecl.decl.loc4: %AForwardDecl.type = fn_decl @AForwardDecl [template = constants.%AForwardDecl] {}
// CHECK:STDOUT:   %AForwardDecl.decl.loc6: %AForwardDecl.type = fn_decl @AForwardDecl [template = constants.%AForwardDecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AForwardDecl() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BForwardDecl.type: type = fn_type @BForwardDecl [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BForwardDecl: %BForwardDecl.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .BForwardDecl = %BForwardDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %BForwardDecl.decl: %BForwardDecl.type = fn_decl @BForwardDecl [template = constants.%BForwardDecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BForwardDecl();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BForwardDecl.type: type = fn_type @BForwardDecl [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BForwardDecl: %BForwardDecl.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AForwardDecl = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .BForwardDecl = %BForwardDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %BForwardDecl.type = import_ref ir0, inst+2, loaded [template = constants.%BForwardDecl]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %BForwardDecl.decl: %BForwardDecl.type = fn_decl @BForwardDecl [template = constants.%BForwardDecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BForwardDecl();
// CHECK:STDOUT:
