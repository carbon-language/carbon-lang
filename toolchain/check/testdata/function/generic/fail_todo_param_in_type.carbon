// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon

// CHECK:STDERR: fail_todo_param_in_type.carbon:[[@LINE+3]]:24: error: semantics TODO: `symbolic array bound` [SemanticsTodo]
// CHECK:STDERR: fn F(N:! i32, a: [i32; N]*);
// CHECK:STDERR:                        ^
fn F(N:! i32, a: [i32; N]*);

// CHECK:STDOUT: --- fail_todo_param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     import Core//prelude/operators/index
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc14_6.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc14_6.1, runtime_param<invalid> [symbolic = %N.patt.loc14_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc14_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_10.1: type = value_of_initializer %int.make_type_32.loc14_10 [template = i32]
// CHECK:STDOUT:     %.loc14_10.2: type = converted %int.make_type_32.loc14_10, %.loc14_10.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc14_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %N.ref: i32 = name_ref N, %N.loc14_6.1 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:     %.loc14_19.1: type = value_of_initializer %int.make_type_32.loc14_19 [template = i32]
// CHECK:STDOUT:     %.loc14_19.2: type = converted %int.make_type_32.loc14_19, %.loc14_19.1 [template = i32]
// CHECK:STDOUT:     %.loc14_25: type = array_type %N.ref, i32 [template = <error>]
// CHECK:STDOUT:     %.loc14_26: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_6.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc14_6.1: i32) {
// CHECK:STDOUT:   %N.loc14_6.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc14_6.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: i32, %a.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc14_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc14_6.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
