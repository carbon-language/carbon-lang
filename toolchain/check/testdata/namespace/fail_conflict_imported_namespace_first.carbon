// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_first.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_conflict_imported_namespace_first.carbon

// --- namespace.carbon

package Example library "namespace";

namespace NS;

// --- fail_conflict.carbon

package Example;

import library "namespace";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+9]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "namespace";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: namespace.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
fn NS();

fn NS.Foo();

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: %.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref Example//namespace, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref, [template] {
// CHECK:STDOUT:     .Foo = file.%Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.2] {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
