// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/fail_no_conversion.carbon

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+6]]:21: error: cannot convert from `Core.IntLiteral` to `(i32, i32)` with `as` [ExplicitAsConversionFailure]
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:21: note: type `Core.IntLiteral` does not implement interface `As((i32, i32))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let n: (i32, i32) = 1 as (i32, i32);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
let n: (i32, i32) = 1 as (i32, i32);

// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc17_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17_9: init type = call constants.%Int(%.loc17_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17_14: init type = call constants.%Int(%.loc17_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc17_9, %int.make_type_signed.loc17_14)
// CHECK:STDOUT:   %.loc17_17.2: type = value_of_initializer %int.make_type_signed.loc17_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.3: type = converted %int.make_type_signed.loc17_9, %.loc17_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.4: type = value_of_initializer %int.make_type_signed.loc17_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.5: type = converted %int.make_type_signed.loc17_14, %.loc17_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.6: type = converted %.loc17_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_21: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_27: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17_27: init type = call constants.%Int(%.loc17_27) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17_32: init type = call constants.%Int(%.loc17_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_35.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc17_27, %int.make_type_signed.loc17_32)
// CHECK:STDOUT:   %.loc17_35.2: type = value_of_initializer %int.make_type_signed.loc17_27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_35.3: type = converted %int.make_type_signed.loc17_27, %.loc17_35.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_35.4: type = value_of_initializer %int.make_type_signed.loc17_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_35.5: type = converted %int.make_type_signed.loc17_32, %.loc17_35.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_35.6: type = converted %.loc17_35.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc17_23.1: %tuple.type.2 = converted %.loc17_21, <error> [template = <error>]
// CHECK:STDOUT:   %.loc17_23.2: %i32 = tuple_access %.loc17_23.1, element0 [template = <error>]
// CHECK:STDOUT:   %n: %tuple.type.2 = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
