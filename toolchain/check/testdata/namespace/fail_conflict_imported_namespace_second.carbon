// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_second.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_second.carbon

// --- fn.carbon

package Example library "[[@TEST_NAME]]";

fn NS();

// --- fail_conflict.carbon

package Example;

import library "fn";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+8]]:11: error: duplicate name `NS` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: namespace NS;
// CHECK:STDERR:           ^~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: fn.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+8]]:4: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: fn NS.Foo();
// CHECK:STDERR:    ^~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: fn.carbon:4:1: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn NS.Foo();

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS.type: type = fn_type @NS [concrete]
// CHECK:STDOUT:   %NS: %NS.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .NS = %NS.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS.decl: %NS.type = fn_decl @NS [concrete = constants.%NS] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS.type: type = fn_type @NS [concrete]
// CHECK:STDOUT:   %NS: %NS.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Example.NS: %NS.type = import_ref Example//fn, NS, loaded [concrete = constants.%NS]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS = imports.%Example.NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %.loc14: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS() [from "fn.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
