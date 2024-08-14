// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuples/fail_non_tuple_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuples/fail_non_tuple_index.carbon

fn Main() {
  var non_tuple: [i32; 1] = (1,);
  // CHECK:STDERR: fail_non_tuple_index.carbon:[[@LINE+3]]:20: ERROR: Type `[i32; 1]` does not support tuple indexing. Only tuples can be indexed that way.
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_24: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_19.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_19.2: type = converted %int.make_type_32.loc12, %.loc12_19.1 [template = i32]
// CHECK:STDOUT:   %.loc12_25: type = array_type %.loc12_24, i32 [template = constants.%.3]
// CHECK:STDOUT:   %non_tuple.var: ref %.3 = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %.3 = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %.loc12_30: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_32.1: %.5 = tuple_literal (%.loc12_30)
// CHECK:STDOUT:   %.loc12_32.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_32.3: ref i32 = array_index %non_tuple.var, %.loc12_32.2
// CHECK:STDOUT:   %.loc12_32.4: init i32 = initialize_from %.loc12_30 to %.loc12_32.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_32.5: init %.3 = array_init (%.loc12_32.4) to %non_tuple.var [template = constants.%array]
// CHECK:STDOUT:   %.loc12_33: init %.3 = converted %.loc12_32.1, %.loc12_32.5 [template = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc12_33
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_14.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_14.2: type = converted %int.make_type_32.loc16, %.loc16_14.1 [template = i32]
// CHECK:STDOUT:   %first.var: ref i32 = var first
// CHECK:STDOUT:   %first: ref i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %.3 = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %.loc16_30: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
