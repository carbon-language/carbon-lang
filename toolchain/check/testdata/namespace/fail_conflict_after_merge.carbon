// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:4-5: error: duplicate name `NS` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_conflict.carbon:[[@LINE-7]]:1 [InImport]
// CHECK:STDERR: namespace.carbon:4:1-13: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
fn NS();

// The second conflict diagnostic should still point at the earlier declaration;
// we don't move it.
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+4]]:4-5: error: duplicate name `NS` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_conflict.carbon:[[@LINE-17]]:1 [InImport]
// CHECK:STDERR: namespace.carbon:4:1-13: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
fn NS();

// CHECK:STDOUT: --- namespace.carbon
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
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NS.type.20a02b.1: type = fn_type @NS.loc14 [concrete]
// CHECK:STDOUT:   %NS.bab1af.1: %NS.type.20a02b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NS.type.20a02b.2: type = fn_type @NS.loc24 [concrete]
// CHECK:STDOUT:   %NS.bab1af.2: %NS.type.20a02b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Example.NS: <namespace> = import_ref Example//namespace, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Example.NS, [concrete] {}
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NS.loc8: <namespace> = namespace imports.%Example.NS, [concrete] {}
// CHECK:STDOUT:   %NS.decl.loc14: %NS.type.20a02b.1 = fn_decl @NS.loc14 [concrete = constants.%NS.bab1af.1] {} {}
// CHECK:STDOUT:   %NS.loc18: <namespace> = namespace imports.%Example.NS, [concrete] {}
// CHECK:STDOUT:   %NS.decl.loc24: %NS.type.20a02b.2 = fn_decl @NS.loc24 [concrete = constants.%NS.bab1af.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS.loc14();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS.loc24();
// CHECK:STDOUT:
