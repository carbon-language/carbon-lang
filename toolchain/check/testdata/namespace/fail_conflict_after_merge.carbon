// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_conflict_after_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_conflict_after_merge.carbon

// --- namespace.carbon

package Example library "[[@TEST_NAME]]";

namespace NS;

// --- fail_conflict.carbon

package Example;

import library "namespace";

// The conflict diagnostics should point at this declaration rather than the
// imported declaration.
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+8]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-9]]:1: in import [InImport]
// CHECK:STDERR: namespace.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn NS();

// The second conflict diagnostic should still point at the earlier declaration;
// we don't move it.
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+7]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-23]]:1: in import [InImport]
// CHECK:STDERR: namespace.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
fn NS();

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.2: %.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref Example//namespace, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref, [template] {}
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %NS.loc8: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.decl.loc18: %.type.1 = fn_decl @.1 [template = constants.%.1] {} {}
// CHECK:STDOUT:   %NS.loc22: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.decl.loc31: %.type.2 = fn_decl @.2 [template = constants.%.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2();
// CHECK:STDOUT:
