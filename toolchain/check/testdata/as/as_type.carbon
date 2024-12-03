// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/as_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/as_type.carbon

let t: type = (i32, i32) as type;

// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
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
// CHECK:STDOUT:     .t = @__global_init.%t
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc11_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_16: init type = call constants.%Int(%int_32.loc11_16) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_21: init type = call constants.%Int(%int_32.loc11_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_24: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_16, %int.make_type_signed.loc11_21)
// CHECK:STDOUT:   %.loc11_26.1: type = value_of_initializer %int.make_type_signed.loc11_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.2: type = converted %int.make_type_signed.loc11_16, %.loc11_26.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.3: type = value_of_initializer %int.make_type_signed.loc11_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.4: type = converted %int.make_type_signed.loc11_21, %.loc11_26.3 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.5: type = converted %.loc11_24, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %t: type = bind_name t, %.loc11_26.5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
