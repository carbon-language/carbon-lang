// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_too_few_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_too_few_element.carbon

// CHECK:STDERR: fail_too_few_element.carbon:[[@LINE+3]]:21: error: cannot initialize tuple of 2 elements from tuple with 1 element [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: (i32, i32) = (2, );
// CHECK:STDERR:                     ^~~~~
var x: (i32, i32) = (2, );

// CHECK:STDOUT: --- fail_too_few_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc14_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_9: init type = call constants.%Int(%int_32.loc14_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_14: init type = call constants.%Int(%int_32.loc14_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_9, %int.make_type_signed.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_signed.loc14_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_signed.loc14_9, %.loc14_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_signed.loc14_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_signed.loc14_14, %.loc14_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc14: %tuple.type.3 = tuple_literal (%int_2)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
