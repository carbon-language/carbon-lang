// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_nested.carbon

// --- fn.carbon

package Other;
namespace Nested;

// --- fail_conflict.carbon

import Other;
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+7]]:11: error: duplicate name `Other` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: namespace Other;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-4]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: import Other;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
namespace Other;
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+8]]:10: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: fn Other.Nested.F();
// CHECK:STDERR:          ^~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-12]]:1: in import [InImport]
// CHECK:STDERR: fn.carbon:3:1: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR: namespace Nested;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Other.Nested.F();

// CHECK:STDOUT: --- fn.carbon
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
// CHECK:STDOUT:     .Nested = %Nested
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Nested: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .Nested = %Nested
// CHECK:STDOUT:     import Other//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.Nested: <namespace> = import_ref Other//default, Nested, loaded
// CHECK:STDOUT:   %Nested: <namespace> = namespace %Other.Nested, [concrete] {
// CHECK:STDOUT:     .F = file.%F.decl
// CHECK:STDOUT:     import Other//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Nested = imports.%Nested
// CHECK:STDOUT:     import Other//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
