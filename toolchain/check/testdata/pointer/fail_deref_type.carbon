// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_deref_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_deref_type.carbon

// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+7]]:8: ERROR: Cannot dereference operand of non-pointer type `type`.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+4]]:8: To form a pointer type, write the `*` after the pointee type.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var p: *i32;
// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+3]]:12: ERROR: Cannot apply `->` operator to non-pointer type `type`.
// CHECK:STDERR: var p2: i32->foo;
// CHECK:STDERR:            ^~
var p2: i32->foo;

// CHECK:STDOUT: --- fail_deref_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:     .p2 = %p2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_9.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_9.2: type = converted %int.make_type_32.loc18, %.loc18_9.1 [template = i32]
// CHECK:STDOUT:   %.loc18_8: ref <error> = deref %.loc18_9.2
// CHECK:STDOUT:   %p.var: ref <error> = var p
// CHECK:STDOUT:   %p: ref <error> = bind_name p, %p.var
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_9.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_9.2: type = converted %int.make_type_32.loc22, %.loc22_9.1 [template = i32]
// CHECK:STDOUT:   %.loc22_12: ref <error> = deref %.loc22_9.2
// CHECK:STDOUT:   %p2.var: ref <error> = var p2
// CHECK:STDOUT:   %p2: ref <error> = bind_name p2, %p2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
